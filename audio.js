/**
 * Driver Drowsiness Detection System - Audio & Sound Alert Module
 * Uses Web Audio API for custom alert tones and Web Speech API for voice warnings.
 */

class DrowsinessAudioEngine {
    constructor() {
        this.audioCtx = null;
        this.alarmOscillator = null;
        this.alarmGain = null;
        this.isPlayingAlarm = false;
        this.alarmInterval = null;
        this.lastSpeechTime = 0;
        this.speechCooldown = 4000; // Minimum ms between voice warnings
        this.alarmVolume = 0.8;
        this.soundType = 'siren'; // 'siren', 'pulse', 'beep'
    }

    init() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    }

    setVolume(volume) {
        this.alarmVolume = parseFloat(volume);
        if (this.alarmGain && this.audioCtx) {
            this.alarmGain.gain.setTargetAtTime(this.alarmVolume, this.audioCtx.currentTime, 0.05);
        }
    }

    setSoundType(type) {
        this.soundType = type;
    }

    startAlarm() {
        this.init();
        if (this.isPlayingAlarm) return;
        this.isPlayingAlarm = true;

        const now = this.audioCtx.currentTime;

        if (this.soundType === 'siren') {
            this.startSirenAlarm(now);
        } else if (this.soundType === 'pulse') {
            this.startPulseAlarm();
        } else {
            this.startBeepAlarm();
        }
    }

    startSirenAlarm(startTime) {
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(440, startTime);
        
        // Frequency sweep for siren effect (440Hz -> 880Hz -> 440Hz repeating)
        const sweepTime = 0.4;
        osc.frequency.linearRampToValueAtTime(880, startTime + sweepTime / 2);
        osc.frequency.linearRampToValueAtTime(440, startTime + sweepTime);

        this.alarmInterval = setInterval(() => {
            if (!this.audioCtx || !this.isPlayingAlarm) return;
            const t = this.audioCtx.currentTime;
            osc.frequency.setValueAtTime(440, t);
            osc.frequency.linearRampToValueAtTime(880, t + 0.2);
            osc.frequency.linearRampToValueAtTime(440, t + 0.4);
        }, 400);

        gain.gain.setValueAtTime(this.alarmVolume, startTime);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start(startTime);

        this.alarmOscillator = osc;
        this.alarmGain = gain;
    }

    startPulseAlarm() {
        let toggle = false;
        this.alarmInterval = setInterval(() => {
            if (!this.isPlayingAlarm) return;
            this.playShortBeep(toggle ? 900 : 600, 0.15);
            toggle = !toggle;
        }, 200);
    }

    startBeepAlarm() {
        this.alarmInterval = setInterval(() => {
            if (!this.isPlayingAlarm) return;
            this.playShortBeep(800, 0.1);
        }, 500);
    }

    playShortBeep(freq = 600, duration = 0.1) {
        this.init();
        if (!this.audioCtx) return;

        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);

        gain.gain.setValueAtTime(this.alarmVolume * 0.7, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + duration);

        osc.connect(gain);
        gain.connect(this.audioCtx.destination);

        osc.start();
        osc.stop(this.audioCtx.currentTime + duration);
    }

    stopAlarm() {
        this.isPlayingAlarm = false;
        if (this.alarmInterval) {
            clearInterval(this.alarmInterval);
            this.alarmInterval = null;
        }

        if (this.alarmOscillator) {
            try {
                this.alarmOscillator.stop();
                this.alarmOscillator.disconnect();
            } catch (e) {}
            this.alarmOscillator = null;
        }

        if (this.alarmGain) {
            try {
                this.alarmGain.disconnect();
            } catch (e) {}
            this.alarmGain = null;
        }
    }

    speakVoiceWarning(message) {
        const now = Date.now();
        if (now - this.lastSpeechTime < this.speechCooldown) return;
        this.lastSpeechTime = now;

        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 1.1;
            utterance.pitch = 1.2;
            utterance.volume = Math.min(1, this.alarmVolume + 0.2);
            
            const voices = window.speechSynthesis.getVoices();
            const engVoice = voices.find(v => v.lang.includes('en') && v.name.includes('Google')) || voices.find(v => v.lang.includes('en'));
            if (engVoice) utterance.voice = engVoice;

            window.speechSynthesis.speak(utterance);
        }
    }
}

window.audioEngine = new DrowsinessAudioEngine();
