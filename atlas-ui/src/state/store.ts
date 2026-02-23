import { create } from 'zustand';
import type { AtlasStore } from '../types/index';

const MAX_HISTORY = 20;
const MAX_EVENTS = 50;
export const PRIVACY_STORAGE_KEY = 'atlas-privacy-mode';

export const useAtlasStore = create<AtlasStore>((set) => ({
  status: 'idle',
  transcript: '',
  response: '',
  isConnected: false,
  audioAnalysis: { volume: 0, frequency: 0, isActive: false, waveform: [] },
  privacyMode: typeof window !== 'undefined' && localStorage.getItem(PRIVACY_STORAGE_KEY) === 'true',
  textInput: '',
  media: null,
  conversationHistory: [],
  systemEvents: [],

  setStatus: (status) => set({ status }),
  setTranscript: (transcript) => set({ transcript }),
  setResponse: (response) => set({ response }),
  setConnected: (isConnected) => set({ isConnected }),
  setAudioAnalysis: (audioAnalysis) => set({ audioAnalysis }),

  setPrivacyMode: (enabled) => {
    localStorage.setItem(PRIVACY_STORAGE_KEY, String(enabled));
    set({ privacyMode: enabled });
  },

  setTextInput: (textInput) => set({ textInput }),
  setMedia: (media) => set({ media }),

  addConversationTurn: (role, text) => set((state) => ({
    conversationHistory: [...state.conversationHistory, { role, text }].slice(-MAX_HISTORY),
  })),

  addSystemEvent: (event) => set((state) => ({
    systemEvents: [...state.systemEvents.slice(-(MAX_EVENTS - 1)), event],
  })),

  reset: () => set({
    status: 'idle',
    transcript: '',
    response: '',
    audioAnalysis: { volume: 0, frequency: 0, isActive: false, waveform: [] },
    media: null,
  })
}));
