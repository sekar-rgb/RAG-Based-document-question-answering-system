import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, 
  UploadCloud, 
  FileText, 
  Trash2, 
  Sliders, 
  Database, 
  Loader2, 
  X, 
  ChevronRight, 
  ChevronLeft,
  MessageSquare, 
  Info, 
  CheckCircle2, 
  AlertCircle,
  HelpCircle,
  Cpu,
  RefreshCw,
  Zap,
  Volume2,
  VolumeX,
  Mic,
  MicOff
} from 'lucide-react';

const API_BASE = 'http://localhost:8000';

const fetchWithRetry = async (url, attempts = 8) => {
  let lastError;

  for (let attempt = 0; attempt < attempts; attempt++) {
    try {
      return await fetch(url);
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }

  throw lastError;
};

export default function App() {
  // RAG States
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! I am your RAG Assistant. Ask me anything about your uploaded documents.',
      id: 'welcome'
    }
  ]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Settings States
  const [selectedModel, setSelectedModel] = useState('mistral-nemo:latest');
  const [temperature, setTemperature] = useState(0.2);
  const [topK, setTopK] = useState(5);
  const [systemPrompt, setSystemPrompt] = useState('');
  const [ollamaRunning, setOllamaRunning] = useState(false);
  const [openrouterConfigured, setOpenrouterConfigured] = useState(false);
  const [supportedModels, setSupportedModels] = useState([]);
  const [modelDetails, setModelDetails] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [voiceMode, setVoiceMode] = useState(false);
  const audioRef = useRef(null);

  // Speech Recognition States & Effect
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);

  // Refs to prevent stale closures in async callbacks
  const voiceModeRef = useRef(voiceMode);
  const isListeningRef = useRef(isListening);

  useEffect(() => {
    voiceModeRef.current = voiceMode;
  }, [voiceMode]);

  useEffect(() => {
    isListeningRef.current = isListening;
  }, [isListening]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setQuery(transcript);
        
        // Auto-submit voice query after a short delay
        setTimeout(() => {
          handleSendQueryVoice(transcript);
        }, 500);
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error', event);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
    }
  }, []);

  const toggleListening = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
    } else {
      // Stop current voice playout if any
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      recognitionRef.current.start();
    }
  };

  // Premium ElevenLabs TTS — streams audio & calls onStartCallback(audioDuration) when ready.
  // Falls back to browser SpeechSynthesis if backend fails.
  const speakText = async (text, onStartCallback = null) => {
    try {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current = null;
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }

      // Strip markdown symbols for clean speech
      const cleanText = text.replace(/[#*`_\-\[\]()>~]/g, ' ').replace(/\s+/g, ' ').trim();

      const autoRestartMic = () => {
        if (voiceModeRef.current && recognitionRef.current && !isListeningRef.current) {
          try { recognitionRef.current.start(); } catch (e) { /* already started */ }
        }
      };

      try {
        // Use the streaming endpoint for lowest possible latency
        const response = await fetch(`${API_BASE}/api/tts/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: cleanText })
        });

        if (!response.ok) throw new Error(`ElevenLabs error: ${response.status}`);

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        audioRef.current = audio;

        audio.onended = autoRestartMic;

        // When we have duration info → fire callback so text animation can sync
        audio.onloadedmetadata = () => {
          if (onStartCallback) onStartCallback(audio.duration);
          audio.play().catch(console.error);
        };

        audio.load();

      } catch (backendError) {
        console.warn('ElevenLabs TTS failed — falling back to browser SpeechSynthesis:', backendError);

        if (window.speechSynthesis) {
          const utterance = new SpeechSynthesisUtterance(cleanText);
          utterance.lang = 'en-US';
          utterance.rate = 0.88;   // Slightly slower = more natural & warm
          utterance.pitch = 1.0;
          utterance.volume = 1.0;

          // Pick a natural female voice if available
          const voices = window.speechSynthesis.getVoices();
          const preferred = voices.find(v =>
            v.lang === 'en-US' && (
              v.name.includes('Samantha') ||
              v.name.includes('Google US English') ||
              v.name.includes('Microsoft Zira') ||
              v.name.includes('Female')
            )
          ) || voices.find(v => v.lang === 'en-US');
          if (preferred) utterance.voice = preferred;

          utterance.onend = autoRestartMic;

          // Trigger text animation immediately since we have no duration info
          if (onStartCallback) onStartCallback(null);
          window.speechSynthesis.speak(utterance);
        } else {
          if (onStartCallback) onStartCallback(null);
        }
      }
    } catch (error) {
      console.error('TTS Error:', error);
      if (onStartCallback) onStartCallback(null);
    }
  };

  // Animate text character-by-character at the given interval speed
  const startTextAnimation = (text, messageId, intervalMs = 18) => {
    let index = 0;
    const timer = setInterval(() => {
      setMessages(prev => prev.map(msg =>
        msg.id === messageId ? { ...msg, content: text.slice(0, index + 1) } : msg
      ));
      index++;
      if (index >= text.length) {
        clearInterval(timer);
        setLoading(false);
      }
    }, intervalMs);
  };


  // Documents States
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadSuccess, setUploadSuccess] = useState('');

  // UI Panels
  const [showSettings, setShowSettings] = useState(false);
  const [activeChunks, setActiveChunks] = useState([]);
  const [selectedChunk, setSelectedChunk] = useState(null);
  const [inspectingMsgId, setInspectingMsgId] = useState(null);

  // File Input Ref
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Initial Data Fetch
  useEffect(() => {
    fetchSettings();
    fetchDocuments();
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const fetchSettings = async () => {
    setModelsLoading(true);
    try {
      const res = await fetchWithRetry(`${API_BASE}/api/settings`);
      if (!res.ok) throw new Error(`Settings request failed: HTTP ${res.status}`);
      const data = await res.json();
      setSelectedModel(data.default_model);
      setTopK(data.default_top_k);
      setOllamaRunning(data.ollama_running);
      setOpenrouterConfigured(data.openrouter_configured || false);
      setSupportedModels(data.supported_models || []);
      setModelDetails(data.model_details || []);
    } catch (err) {
      console.error('Error fetching settings:', err);
      setOllamaRunning(false);
      setOpenrouterConfigured(false);
    } finally {
      setModelsLoading(false);
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetchWithRetry(`${API_BASE}/api/documents`);
      if (!res.ok) throw new Error(`Documents request failed: HTTP ${res.status}`);
      const data = await res.json();
      setDocuments(data);
    } catch (err) {
      console.error('Error fetching documents:', err);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    setUploading(true);
    setUploadError('');
    setUploadSuccess('');

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await res.json();
      setUploadSuccess(`Successfully indexed ${files.length} document(s).`);
      fetchDocuments();
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleDeleteDoc = async (name) => {
    if (!confirm(`Are you sure you want to delete ${name}? This will update the search index.`)) return;

    try {
      const res = await fetch(`${API_BASE}/api/documents/${name}`, {
        method: 'DELETE',
      });

      if (!res.ok) throw new Error('Failed to delete document');
      
      setUploadSuccess(`Deleted ${name} and updated index.`);
      fetchDocuments();
      // Reset active chunks if they are from deleted doc
      setActiveChunks([]);
      setInspectingMsgId(null);
    } catch (err) {
      setUploadError(err.message);
    }
  };

  // Orchestrate voice + text animation — synchronized together
  const simulateStreaming = (text, messageId, chunks) => {
    if (voiceModeRef.current) {
      // Start TTS immediately; when audio duration is known → start text animation synced to it
      speakText(text, (audioDuration) => {
        // If we have audio duration, pace text to finish at the same time as speech
        const intervalMs = audioDuration
          ? Math.max(8, Math.min(60, (audioDuration * 1000) / text.length))
          : 18; // fallback speed when no duration
        startTextAnimation(text, messageId, intervalMs);
      });
    } else {
      startTextAnimation(text, messageId, 18);
    }
  };

  const handleSendQueryVoice = async (spokenText) => {
    if (!spokenText.trim() || loading) return;

    // Pause any playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    const userMessage = {
      role: 'user',
      content: spokenText,
      id: `user-${Date.now()}`
    };

    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);

    const assistantMsgId = `assistant-${Date.now()}`;
    const placeholderMessage = {
      role: 'assistant',
      content: '',
      id: assistantMsgId,
      loading: true
    };
    
    setMessages(prev => [...prev, placeholderMessage]);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: spokenText,
          model: selectedModel,
          top_k: topK,
          temperature: parseFloat(temperature),
          system_prompt: systemPrompt || null
        })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Query request failed');
      }

      const data = await res.json();

      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMsgId) {
          return { 
            ...msg, 
            loading: false,
            chunks: data.context_chunks 
          };
        }
        return msg;
      }));

      simulateStreaming(data.answer, assistantMsgId, data.context_chunks);
      setActiveChunks(data.context_chunks);
      setInspectingMsgId(assistantMsgId);
      setSelectedChunk(data.context_chunks[0] || null);

    } catch (err) {
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMsgId) {
          return { 
            ...msg, 
            loading: false,
            content: `Error: ${err.message}`, 
            error: true 
          };
        }
        return msg;
      }));
      setLoading(false);
    }
  };

  const handleSendQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    // Pause any playing audio
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    const userMessage = {
      role: 'user',
      content: query,
      id: `user-${Date.now()}`
    };

    setMessages(prev => [...prev, userMessage]);
    const currentQuery = query;
    setQuery('');
    setLoading(true);

    const assistantMsgId = `assistant-${Date.now()}`;
    const placeholderMessage = {
      role: 'assistant',
      content: '',
      id: assistantMsgId,
      loading: true
    };
    
    setMessages(prev => [...prev, placeholderMessage]);

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          model: selectedModel,
          top_k: topK,
          temperature: parseFloat(temperature),
          system_prompt: systemPrompt || null
        })
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Query request failed');
      }

      const data = await res.json();

      // Remove loading status
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMsgId) {
          return { 
            ...msg, 
            loading: false,
            chunks: data.context_chunks 
          };
        }
        return msg;
      }));

      // Simulate streaming the answer
      simulateStreaming(data.answer, assistantMsgId, data.context_chunks);
      
      // Auto display retrieved chunks in side panel
      setActiveChunks(data.context_chunks);
      setInspectingMsgId(assistantMsgId);
      setSelectedChunk(data.context_chunks[0] || null);

    } catch (err) {
      setMessages(prev => prev.map(msg => {
        if (msg.id === assistantMsgId) {
          return { 
            ...msg, 
            loading: false,
            content: `Error: ${err.message}`, 
            error: true 
          };
        }
        return msg;
      }));
      setLoading(false);
    }
  };

  const inspectCitations = (msg) => {
    if (!msg.chunks || msg.chunks.length === 0) return;
    setActiveChunks(msg.chunks);
    setInspectingMsgId(msg.id);
    setSelectedChunk(msg.chunks[0] || null);
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="dashboard-container">
      {/* 1. Sidebar - Document Manager & Quick settings toggle */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="sidebar-title">
            <Database className="text-accent-cyan" size={22} />
            RAG Studio
          </h1>
          <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '6px' }}>
            <span className={`badge ${ollamaRunning ? 'badge-success' : 'badge-error'}`}>
              <Zap size={9} style={{ display: 'inline', marginRight: '3px' }} />
              {ollamaRunning ? 'Ollama Online' : 'Ollama Offline'}
            </span>
            {openrouterConfigured && (
              <span className="badge" style={{ display: 'inline-flex', alignItems: 'center', gap: '3px', padding: '2px 8px', fontSize: '10px', borderRadius: '4px', backgroundColor: '#8b5cf6', color: '#ffffff', fontWeight: 'bold' }}>
                <Cpu size={9} />
                OpenRouter Active
              </span>
            )}
          </div>
        </div>

        <div className="sidebar-content">
          {/* Document Section */}
          <div>
            <div className="sidebar-section-title">Documents</div>
            
            {/* Upload Area */}
            <div className="upload-zone" onClick={handleUploadClick}>
              <UploadCloud style={{ margin: '0 auto 8px auto', display: 'block', color: 'var(--accent-cyan)' }} size={28} />
              <p style={{ fontSize: '13px', fontWeight: '500' }}>
                {uploading ? 'Processing files...' : 'Upload new source'}
              </p>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                Supports .txt, .pdf, .docx
              </p>
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileChange} 
                style={{ display: 'none' }} 
                multiple
                accept=".txt,.pdf,.docx"
              />
            </div>

            {/* Notification indicators */}
            {uploadError && (
              <div style={{ padding: '10px', marginTop: '10px', borderRadius: '6px', background: 'rgba(244, 63, 94, 0.1)', color: 'var(--accent-rose)', fontSize: '12px', display: 'flex', gap: '6px', alignItems: 'center' }}>
                <AlertCircle size={14} style={{ flexShrink: 0 }} />
                <span>{uploadError}</span>
              </div>
            )}
            {uploadSuccess && (
              <div style={{ padding: '10px', marginTop: '10px', borderRadius: '6px', background: 'rgba(16, 185, 129, 0.1)', color: '#10b981', fontSize: '12px', display: 'flex', gap: '6px', alignItems: 'center' }}>
                <CheckCircle2 size={14} style={{ flexShrink: 0 }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{uploadSuccess}</span>
              </div>
            )}

            {/* Document List */}
            <div style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '12px', color: 'var(--text-secondary)', fontWeight: '600' }}>
                <span>Uploaded Files</span>
                <span>({documents.length})</span>
              </div>
              
              {documents.length === 0 ? (
                <div style={{ padding: '20px 0', textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
                  No documents uploaded yet.
                </div>
              ) : (
                <div className="doc-list">
                  {documents.map((doc) => (
                    <div className="doc-card glow-hover" key={doc.name}>
                      <div className="doc-info">
                        <FileText size={15} style={{ color: 'var(--accent-cyan)' }} />
                        <div>
                          <p className="doc-name" title={doc.name}>{doc.name}</p>
                          <span className="doc-size">{formatBytes(doc.sizeBytes)}</span>
                        </div>
                      </div>
                      <div className="doc-actions" onClick={() => handleDeleteDoc(doc.name)}>
                        <Trash2 size={14} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            
            {/* Voice Mode Option */}
            <div style={{ marginTop: '16px', background: 'rgba(255, 255, 255, 0.01)', border: '1px solid var(--border-color)', borderRadius: '8px', padding: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {voiceMode ? <Volume2 size={16} style={{ color: 'var(--accent-cyan)' }} /> : <VolumeX size={16} style={{ color: 'var(--text-muted)' }} />}
                  <span style={{ fontSize: '13px', fontWeight: '600' }}>Interactive Voice Mode</span>
                </div>
                <label className="switch" style={{ position: 'relative', display: 'inline-block', width: '34px', height: '20px', cursor: 'pointer' }}>
                  <input 
                    type="checkbox" 
                    checked={voiceMode} 
                    onChange={(e) => {
                      setVoiceMode(e.target.checked);
                      if (!e.target.checked) {
                        if (audioRef.current) {
                          audioRef.current.pause();
                          audioRef.current = null;
                        }
                        if (isListening && recognitionRef.current) {
                          recognitionRef.current.stop();
                        }
                      }
                    }}
                    style={{ opacity: 0, width: 0, height: 0 }}
                  />
                  <span className="sliderround" style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                    backgroundColor: voiceMode ? 'var(--accent-cyan)' : '#374151',
                    transition: '.4s', borderRadius: '34px'
                  }}>
                    <span style={{
                      position: 'absolute', content: '""', height: '14px', width: '14px', left: '3px', bottom: '3px',
                      backgroundColor: 'white', transition: '.4s', borderRadius: '50%',
                      transform: voiceMode ? 'translateX(14px)' : 'translateX(0)'
                    }} />
                  </span>
                </label>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '6px', lineHeight: '1.4' }}>
                {voiceMode ? '🔊 Active: AI reads responses and automatically listens for follow-up voice commands.' : '🔇 Inactive: Text only. Click the microphone button to use voice manually.'}
              </p>
            </div>
          </div>

          {/* Quick Settings Section */}
          <div style={{ marginTop: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div className="sidebar-section-title" style={{ marginBottom: 0 }}>Model Settings</div>
              <button
                onClick={fetchSettings}
                className="doc-actions"
                style={{ opacity: 0.7 }}
                title="Refresh models"
              >
                <RefreshCw size={13} className={modelsLoading ? 'spin' : ''} />
              </button>
            </div>
            
            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Cpu size={14} /> Select Model
              </label>
              {supportedModels.length === 0 ? (
                <div style={{ padding: '10px', borderRadius: '6px', background: 'rgba(244, 63, 94, 0.1)', color: 'var(--accent-rose)', fontSize: '12px' }}>
                  {ollamaRunning === false ? 'Ollama not running. Start with: ollama serve' : 'No models found.'}
                </div>
              ) : (
                <>
                  <select 
                    className="form-select" 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value)}
                  >
                    {supportedModels.map(m => {
                      const isLocal = !m.includes('/');
                      const detail = modelDetails.find(d => d.name === m);
                      const sizeGB = detail ? (detail.size / 1e9).toFixed(1) + ' GB' : '';
                      const providerLabel = isLocal ? 'Local Ollama' : 'OpenRouter';
                      return (
                        <option key={m} value={m}>
                          [{providerLabel}] {m}{sizeGB ? ` (${sizeGB})` : ''}
                        </option>
                      );
                    })}
                  </select>
                  {!ollamaRunning && selectedModel && !selectedModel.includes('/') && (
                    <div style={{ padding: '6px 8px', borderRadius: '4px', background: 'rgba(244, 63, 94, 0.05)', color: 'var(--accent-rose)', fontSize: '11px', marginTop: '6px' }}>
                      ⚠️ Selected model is offline. Run 'ollama serve' or select an OpenRouter model.
                    </div>
                  )}
                </>
              )}
            </div>

            <button 
              className="btn btn-secondary" 
              style={{ width: '100%', display: 'flex', justifyContent: 'center' }}
              onClick={() => setShowSettings(!showSettings)}
            >
              <Sliders size={15} />
              {showSettings ? 'Hide Details' : 'Configure Parameters'}
            </button>
          </div>
        </div>
      </aside>

      {/* 2. Main Chat Interface */}
      <main className="main-content">
        <div className="chat-container">
          <header className="chat-header">
            <div>
              <h2 style={{ fontSize: '16px', fontWeight: '600' }}>Conversation Playground</h2>
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                Using: <span style={{ color: 'var(--accent-cyan)' }}>{selectedModel}</span> · FAISS Vector Retrieval
              </p>
            </div>
            {activeChunks.length > 0 && (
              <button 
                className="btn btn-secondary" 
                style={{ padding: '6px 12px', fontSize: '12px' }}
                onClick={() => setInspectingMsgId(inspectingMsgId ? null : 'active')}
              >
                <Info size={14} />
                {inspectingMsgId ? 'Hide Citations' : 'Inspect Citations'}
              </button>
            )}
          </header>

          {/* Chat Bubble List */}
          <div className="chat-messages">
            {messages.map((msg) => (
              <div className={`message-bubble ${msg.role}`} key={msg.id}>
                <div className="avatar">
                  {msg.role === 'user' ? 'U' : 'AI'}
                </div>
                <div className="message-content">
                  {msg.loading ? (
                    <div className="typing-dots">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  ) : (
                    <div>
                      <p style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</p>
                      
                      {/* Citations Indicator */}
                      {msg.chunks && msg.chunks.length > 0 && (
                        <div style={{ marginTop: '10px', display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
                          <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                            <FileText size={11} /> Sources:
                          </span>
                          {msg.chunks.map((c, i) => (
                            <button
                              key={i}
                              onClick={() => inspectCitations(msg)}
                              className="badge badge-success"
                              style={{ 
                                cursor: 'pointer', 
                                border: 'none', 
                                display: 'inline-flex',
                                alignItems: 'center', 
                                gap: '3px'
                              }}
                            >
                              {c.source} <ChevronRight size={10} />
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Chat Query Input */}
          <div className="chat-input-area">
            <form onSubmit={handleSendQuery}>
              <div className="input-wrapper">
                <input 
                  type="text" 
                  className="chat-input"
                  placeholder="Ask a question about the active directory files..." 
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={loading}
                />
                <button
                  type="button"
                  onClick={toggleListening}
                  className={`btn ${isListening ? 'btn-danger pulse-active' : 'btn-secondary'}`}
                  style={{
                    height: '40px',
                    width: '40px',
                    padding: 0,
                    borderRadius: '8px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: isListening ? '#f43f5e' : 'var(--bg-input)',
                    borderColor: isListening ? '#f43f5e' : 'var(--border-color)',
                    color: isListening ? '#ffffff' : 'var(--text-secondary)',
                    marginRight: '6px'
                  }}
                  title={isListening ? "Listening... Click to stop" : "Speak your query"}
                >
                  {isListening ? <Mic size={16} /> : <MicOff size={16} />}
                </button>
                <button 
                  type="submit" 
                  className="btn btn-primary" 
                  style={{ height: '40px', width: '40px', padding: 0 }}
                  disabled={loading || !query.trim()}
                >
                  {loading ? <Loader2 className="animate-spin" size={16} /> : <Send size={16} />}
                </button>
              </div>
            </form>
          </div>
        </div>
      </main>

      {/* 3. Advanced Parameters Drawer (Pushes in from right or overlaps) */}
      {showSettings && (
        <div className="side-panel" style={{ borderLeft: '1px solid var(--border-color)', borderRight: 'none' }}>
          <div className="side-panel-header">
            <h3 style={{ fontSize: '15px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sliders size={16} style={{ color: 'var(--accent-cyan)' }} />
              Advanced Settings
            </h3>
            <button className="doc-actions" onClick={() => setShowSettings(false)}>
              <X size={18} />
            </button>
          </div>
          
          <div className="side-panel-content">
            <div className="form-group">
              <div className="slider-group">
                <label className="form-label">Temperature (Creative vs Focused)</label>
                <span className="slider-val">{temperature}</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="0.0" 
                max="1.0" 
                step="0.1" 
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
              />
            </div>

            <div className="form-group">
              <div className="slider-group">
                <label className="form-label">Top-K Context Chunks</label>
                <span className="slider-val">{topK}</span>
              </div>
              <input 
                type="range" 
                className="slider-input" 
                min="1" 
                max="10" 
                step="1" 
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">System Instruction Override</label>
              <textarea 
                className="form-input" 
                style={{ height: '140px', resize: 'none', fontFamily: 'inherit', fontSize: '13px' }}
                placeholder="e.g. You are a strict data-science analyst. Output answers as bullets."
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
              />
            </div>

            <div style={{ marginTop: 'auto', background: 'rgba(255, 255, 255, 0.02)', padding: '12px', borderRadius: '8px', border: '1px solid var(--border-color)', fontSize: '12px', color: 'var(--text-secondary)' }}>
              <h4 style={{ fontWeight: '600', color: 'var(--text-primary)', marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Info size={13} style={{ color: 'var(--accent-cyan)' }} /> Prompting Tip
              </h4>
              Lower temperature values (0.0 to 0.2) yield more precise responses strictly matching context citations, while higher values yield creative summary answers.
            </div>
          </div>
        </div>
      )}

      {/* 4. Citation Inspector Toggle Handle (when hidden but activeChunks exist) */}
      {!inspectingMsgId && activeChunks.length > 0 && (
        <button 
          onClick={() => setInspectingMsgId('active')}
          className="glow-hover"
          style={{
            position: 'absolute',
            right: '0',
            top: '50%',
            transform: 'translateY(-50%)',
            writingMode: 'vertical-rl',
            padding: '16px 8px',
            cursor: 'pointer',
            zIndex: 10,
            color: 'var(--accent-cyan)',
            backgroundColor: 'var(--bg-sidebar)',
            border: '1px solid var(--border-color)',
            borderRight: 'none',
            borderTopLeftRadius: '8px',
            borderBottomLeftRadius: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '12px',
            fontWeight: '600'
          }}
        >
          <ChevronLeft size={16} style={{ transform: 'rotate(-90deg)', margin: '4px 0' }} />
          Show Citations
        </button>
      )}

      {/* 5. Citation Inspector (Slides in from the right) */}
      {inspectingMsgId && activeChunks.length > 0 && (
        <div className="side-panel">
          <div className="side-panel-header">
            <h3 style={{ fontSize: '15px', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <FileText size={16} style={{ color: 'var(--accent-cyan)' }} />
              Retrieved Chunks ({activeChunks.length})
            </h3>
            <button className="doc-actions" onClick={() => setInspectingMsgId(null)}>
              <X size={18} />
            </button>
          </div>

          <div className="side-panel-content" style={{ display: 'flex', flexDirection: 'row', padding: 0, overflow: 'hidden' }}>
            {/* Chunk selector list */}
            <div style={{ width: '160px', borderRight: '1px solid var(--border-color)', height: '100%', overflowY: 'auto', padding: '12px' }}>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px', fontWeight: '600' }}>Select Chunk</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {activeChunks.map((chunk, index) => (
                  <div
                    key={index}
                    className={`chunk-card ${selectedChunk === chunk ? 'active' : ''}`}
                    style={{ padding: '8px 10px', fontSize: '12px' }}
                    onClick={() => setSelectedChunk(chunk)}
                  >
                    <div style={{ fontWeight: '600', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {chunk.source}
                    </div>
                    <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Score: {chunk.score.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Chunk detailed text view */}
            <div style={{ flexGrow: 1, padding: '20px', height: '100%', overflowY: 'auto', display: 'flex', flexDirection: 'column' }}>
              {selectedChunk ? (
                <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', marginBottom: '12px' }}>
                    <div>
                      <h4 style={{ fontSize: '14px', fontWeight: '600', color: 'var(--accent-cyan)' }}>{selectedChunk.source}</h4>
                      <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>Cosine Distance Score: {selectedChunk.score.toFixed(4)}</p>
                    </div>
                  </div>
                  <div style={{ flexGrow: 1, overflowY: 'auto', background: 'rgba(255, 255, 255, 0.01)', border: '1px solid var(--border-color)', borderRadius: '6px', padding: '12px', fontSize: '13px', lineHeight: '1.6', whiteSpace: 'pre-wrap', color: 'var(--text-primary)' }}>
                    {selectedChunk.text}
                  </div>
                </div>
              ) : (
                <div className="empty-state">
                  <HelpCircle className="empty-state-icon" />
                  <p style={{ fontSize: '13px' }}>Select a chunk card from the left panel to inspect raw text.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
