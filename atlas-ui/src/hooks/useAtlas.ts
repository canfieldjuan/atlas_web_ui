import { useEffect, useRef, useCallback } from 'react';
import { useAtlasStore, PRIVACY_STORAGE_KEY } from '../state/store';
import { getWebSocketUrl } from '../config/connection';

export const useAtlas = (url: string = getWebSocketUrl()) => {
  const {
    setStatus, setTranscript, setResponse, setConnected, setAudioAnalysis,
    setMedia, addConversationTurn, addSystemEvent,
  } = useAtlasStore();
  const ws = useRef<WebSocket | null>(null);
  const currentAudio = useRef<HTMLAudioElement | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const animationFrame = useRef<number | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const readingTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000;

  const connect = useCallback(() => {
    // Don't create new connection if one exists and is open or connecting
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    // Close any existing connection in closing state
    if (ws.current) {
      ws.current.close();
    }

    ws.current = new WebSocket(url);

    ws.current.onopen = () => {
      console.log('Connected to Atlas');
      setConnected(true);
      reconnectAttempts.current = 0;

      // Sync privacy mode from localStorage to server
      const privacyStored = localStorage.getItem(PRIVACY_STORAGE_KEY) === 'true';
      if (privacyStored && ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ command: 'set_privacy', enabled: true }));
      }
    };

    ws.current.onclose = () => {
      console.log('Disconnected from Atlas');
      setConnected(false);

      // Auto-reconnect with exponential backoff
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts.current);
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current + 1})`);
        reconnectTimeout.current = setTimeout(() => {
          reconnectAttempts.current++;
          connect();
        }, delay);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WS Message:', data);

        switch (data.state) {
          case 'idle':
            setStatus('idle');
            break;
          case 'listening':
            setStatus('idle');
            break;
          case 'wake_detected':
          case 'recording':
            setStatus('listening');
            break;
          case 'transcribing':
          case 'processing':
          case 'executing':
            setStatus('processing');
            break;
          case 'responding':
            setStatus('speaking');
            break;
          case 'error':
            setStatus('error');
            console.error('Atlas Error:', data.message);
            break;
          case 'transcript':
            setTranscript(data.text);
            break;
          case 'response': {
            // Add user turn from transcript for voice path
            // (text path already added it in sendText())
            const currentTranscript = useAtlasStore.getState().transcript;
            if (currentTranscript) {
              const history = useAtlasStore.getState().conversationHistory;
              const lastUser = [...history].reverse().find(t => t.role === 'user');
              if (!lastUser || lastUser.text !== currentTranscript) {
                addConversationTurn('user', currentTranscript);
              }
            }

            setResponse(data.text);
            addConversationTurn('assistant', data.text);
            setTranscript('');

            // Media attachment (camera feed, etc.)
            if (data.media) {
              setMedia(data.media);
            }

            if (data.audio_base64) {
              // Audio response -- play it
              setStatus('speaking');
              try {
                // Stop any currently playing audio
                if (currentAudio.current) {
                  currentAudio.current.pause();
                  currentAudio.current = null;
                }
                if (animationFrame.current) {
                  cancelAnimationFrame(animationFrame.current);
                }

                const binaryString = atob(data.audio_base64);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                  bytes[i] = binaryString.charCodeAt(i);
                }
                const blob = new Blob([bytes], { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(blob);
                const audio = new Audio(audioUrl);
                currentAudio.current = audio;

                // Set up audio analysis
                if (!audioContext.current) {
                  audioContext.current = new AudioContext();
                }
                if (!analyser.current) {
                  analyser.current = audioContext.current.createAnalyser();
                  analyser.current.fftSize = 256;
                }

                const source = audioContext.current.createMediaElementSource(audio);
                source.connect(analyser.current);
                analyser.current.connect(audioContext.current.destination);

                // Start analyzing audio
                const dataArray = new Uint8Array(analyser.current.frequencyBinCount);
                const waveformData = new Uint8Array(128);
                const analyzeAudio = () => {
                  if (!analyser.current || !currentAudio.current) return;

                  analyser.current.getByteFrequencyData(dataArray);
                  analyser.current.getByteTimeDomainData(waveformData);
                  const waveform = Array.from(waveformData).map(v => (v - 128) / 128);

                  const sum = dataArray.reduce((a, b) => a + b, 0);
                  const average = sum / dataArray.length;
                  const volume = Math.min(100, (average / 255) * 100);

                  const maxIndex = dataArray.indexOf(Math.max(...Array.from(dataArray)));
                  const frequency = (maxIndex / dataArray.length) * (audioContext.current?.sampleRate || 44100) / 2;

                  setAudioAnalysis({
                    volume,
                    frequency,
                    isActive: !currentAudio.current.paused && !currentAudio.current.ended,
                    waveform
                  });

                  animationFrame.current = requestAnimationFrame(analyzeAudio);
                };
                analyzeAudio();

                audio.onended = () => {
                  URL.revokeObjectURL(audioUrl);
                  currentAudio.current = null;
                  if (animationFrame.current) {
                    cancelAnimationFrame(animationFrame.current);
                  }
                  setAudioAnalysis({ volume: 0, frequency: 0, isActive: false, waveform: [] });
                  setStatus('idle');
                };
                audio.onerror = (err) => {
                  console.error('Audio playback error:', err);
                  currentAudio.current = null;
                  if (animationFrame.current) {
                    cancelAnimationFrame(animationFrame.current);
                  }
                  setAudioAnalysis({ volume: 0, frequency: 0, isActive: false, waveform: [] });
                  setStatus('idle');
                };
                audio.play().catch(err => console.error('Audio play failed:', err));
              } catch (err) {
                console.error('Audio decode error:', err);
              }
            } else {
              // Text-only response (privacy mode or TTS failure)
              setStatus('reading');
              if (readingTimeout.current) {
                clearTimeout(readingTimeout.current);
              }
              const delay = Math.min(data.text.length * 30, 3000);
              readingTimeout.current = setTimeout(() => {
                setStatus('idle');
              }, delay);
            }
            break;
          }

          case 'system_event':
            addSystemEvent({
              id: data.id || String(Date.now()),
              ts: data.ts || new Date().toISOString(),
              category: data.category || 'llm',
              level: data.level || 'info',
              message: data.message || '',
            });
            break;

          default:
            console.log('Unhandled state:', data.state);
        }
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    };

  }, [url, setStatus, setTranscript, setResponse, setConnected, setAudioAnalysis, setMedia, addConversationTurn, addSystemEvent]);

  useEffect(() => {
    connect();

    return () => {
      if (reconnectTimeout.current) {
        clearTimeout(reconnectTimeout.current);
      }
      if (readingTimeout.current) {
        clearTimeout(readingTimeout.current);
      }
      ws.current?.close();
    };
  }, [connect]);

  const sendAudioData = (data: ArrayBuffer) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('Sending audio chunk:', data.byteLength, 'bytes');
      ws.current.send(data);
    } else {
      console.warn('WebSocket not open, state:', ws.current?.readyState);
    }
  };

  const stopRecording = () => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      console.log('Sending stop_recording command');
      ws.current.send(JSON.stringify({ command: 'stop_recording' }));
    }
  };

  const sendText = (text: string) => {
    if (ws.current?.readyState === WebSocket.OPEN && text.trim()) {
      // Add user turn to history
      addConversationTurn('user', text.trim());
      ws.current.send(JSON.stringify({ command: 'send_text', text: text.trim() }));
    }
  };

  const setPrivacy = (enabled: boolean) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ command: 'set_privacy', enabled }));
    }
  };

  return {
    isConnected: useAtlasStore((s) => s.isConnected),
    sendAudioData,
    stopRecording,
    sendText,
    setPrivacy,
    ws: ws.current
  };
};
