import { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from '@google/genai';
import { Upload, FileVideo, FileImage, FileText, Loader2, AlertTriangle, CheckCircle, Info, X, ShieldAlert, ShieldCheck, Activity, BrainCircuit, Fingerprint, ScanSearch, Download, Database, Mic, Radar, Globe } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'motion/react';
import { Radar as RechartsRadar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';

type AnalysisResult = {
  verdict: 'AUTHENTIC' | 'SUSPICIOUS' | 'DEEPFAKE';
  confidenceScore: number;
  vectorScores: {
    visual: number;
    audio: number;
    temporal: number;
    logical: number;
  };
  forensicAnalysis: {
    visualArtifacts: string;
    lightingAndShadows: string;
    anatomyAndPhysics: string;
    temporalConsistency: string;
  };
  riskFactors: string[];
  authenticityIndicators: string[];
  executiveSummary: string;
};

const ANALYSIS_STEPS = [
  "Initializing neural pathways...",
  "Extracting spatial features...",
  "Analyzing lighting & shadow consistency...",
  "Detecting generative artifacts...",
  "Evaluating anatomical structures...",
  "Cross-referencing temporal anomalies...",
  "Compiling forensic report..."
];

export function DeepfakeDetector() {
  const [activeTab, setActiveTab] = useState<'media' | 'text'>('media');
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStep, setAnalysisStep] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hexDump, setHexDump] = useState<string>('');
  const [fileMeta, setFileMeta] = useState<{name: string, size: string, type: string} | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAnalyzing) {
      interval = setInterval(() => {
        setAnalysisStep((prev) => (prev < ANALYSIS_STEPS.length - 1 ? prev + 1 : prev));
      }, 1500);
    } else {
      setAnalysisStep(0);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const processFile = (selectedFile: File) => {
    if (selectedFile.size > 15 * 1024 * 1024) {
      setError('File size must be less than 15MB.');
      return;
    }
    setFile(selectedFile);
    setResult(null);
    setError(null);
    
    setFileMeta({
      name: selectedFile.name,
      size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: selectedFile.type || 'unknown'
    });

    const reader = new FileReader();
    reader.onload = (e) => {
      const buffer = e.target?.result as ArrayBuffer;
      const view = new Uint8Array(buffer);
      let hex = '';
      for(let i=0; i<Math.min(view.length, 256); i++) {
        hex += view[i].toString(16).padStart(2, '0') + ' ';
        if ((i + 1) % 16 === 0) hex += '\n';
      }
      setHexDump(hex.toUpperCase());
    };
    reader.readAsArrayBuffer(selectedFile.slice(0, 256));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) processFile(selectedFile);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) processFile(droppedFile);
  };

  const analyze = async () => {
    if (activeTab === 'media' && !file) return;
    if (activeTab === 'text' && !text.trim()) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || process.env.GEMINI_API_KEY || '' });
      let contents: any;

      if (activeTab === 'media' && file) {
        const base64Data = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve((reader.result as string).split(',')[1]);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });

        contents = {
          parts: [
            {
              inlineData: {
                data: base64Data,
                mimeType: file.type,
              },
            },
            {
              text: "You are an advanced digital forensics AI. Perform a rigorous, multi-vector analysis of the provided media (image, video, or audio) to determine if it is a deepfake, AI-generated, or manipulated. To minimize false positives, require strong evidence of manipulation before classifying as DEEPFAKE. If there are minor anomalies but overall consistency, classify as SUSPICIOUS. If it appears natural, classify as AUTHENTIC. Analyze visual artifacts, lighting, anatomy, noise patterns, and audio anomalies. If the media is audio-only, focus on audio anomalies and logical flow.",
            },
          ],
        };
      } else {
        contents = {
          parts: [
            {
              text: `You are an advanced digital forensics AI. Perform a rigorous analysis of the provided text to determine if it is AI-generated. Look for LLM hallmarks: perplexity, burstiness, repetitive structures, lack of personal voice, or hallucinated facts. To minimize false positives, require strong evidence of generation before classifying as DEEPFAKE. If there are minor anomalies but overall consistency, classify as SUSPICIOUS. If it appears natural, classify as AUTHENTIC. Text: "${text}"`,
            },
          ],
        };
      }

      const response = await ai.models.generateContent({
        model: 'gemini-3.1-pro-preview',
        contents,
        config: {
          tools: [{ googleSearch: {} }],
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              verdict: { type: Type.STRING, enum: ['AUTHENTIC', 'SUSPICIOUS', 'DEEPFAKE'] },
              confidenceScore: { type: Type.NUMBER, description: "Confidence score from 0 to 100." },
              vectorScores: {
                type: Type.OBJECT,
                properties: {
                  visual: { type: Type.NUMBER, description: "Anomaly score for visual artifacts (0-100). Higher means more anomalous. 0 if audio-only." },
                  audio: { type: Type.NUMBER, description: "Anomaly score for audio artifacts (0-100). Higher means more anomalous. 0 if image-only." },
                  temporal: { type: Type.NUMBER, description: "Anomaly score for temporal inconsistencies (0-100). Higher means more anomalous. 0 if static image." },
                  logical: { type: Type.NUMBER, description: "Anomaly score for semantic/logical errors (0-100)." }
                }
              },
              forensicAnalysis: {
                type: Type.OBJECT,
                properties: {
                  visualArtifacts: { type: Type.STRING, description: "Analysis of generative artifacts, blurring, or structural anomalies (or textual structure anomalies)." },
                  lightingAndShadows: { type: Type.STRING, description: "Analysis of lighting, reflections, and shadows (or semantic consistency for text/audio)." },
                  anatomyAndPhysics: { type: Type.STRING, description: "Analysis of anatomical correctness, physics, or logical flow." },
                  temporalConsistency: { type: Type.STRING, description: "Analysis of temporal consistency, flickering, or narrative consistency." },
                }
              },
              riskFactors: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Specific signs of manipulation." },
              authenticityIndicators: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Specific signs of authenticity." },
              executiveSummary: { type: Type.STRING, description: "A concise summary of the findings." }
            },
            required: ['verdict', 'confidenceScore', 'vectorScores', 'forensicAnalysis', 'riskFactors', 'authenticityIndicators', 'executiveSummary'],
          },
        },
      });

      const resultText = response.text;
      if (resultText) {
        setResult(JSON.parse(resultText) as AnalysisResult);
      } else {
        throw new Error("No response from model.");
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred during analysis.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const clearFile = () => {
    setFile(null);
    setResult(null);
    setHexDump('');
    setFileMeta(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const downloadReport = () => {
    if (!result) return;
    const report = {
      timestamp: new Date().toISOString(),
      fileMetadata: fileMeta,
      analysis: result
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case 'DEEPFAKE': return 'text-rose-500 border-rose-500/30 bg-rose-500/10 shadow-[0_0_15px_rgba(244,63,94,0.2)]';
      case 'SUSPICIOUS': return 'text-amber-500 border-amber-500/30 bg-amber-500/10 shadow-[0_0_15px_rgba(245,158,11,0.2)]';
      case 'AUTHENTIC': return 'text-emerald-500 border-emerald-500/30 bg-emerald-500/10 shadow-[0_0_15px_rgba(16,185,129,0.2)]';
      default: return 'text-slate-500 border-slate-500/30 bg-slate-500/10';
    }
  };

  const getVerdictIcon = (verdict: string) => {
    switch (verdict) {
      case 'DEEPFAKE': return <ShieldAlert className="w-8 h-8 text-rose-500" />;
      case 'SUSPICIOUS': return <AlertTriangle className="w-8 h-8 text-amber-500" />;
      case 'AUTHENTIC': return <ShieldCheck className="w-8 h-8 text-emerald-500" />;
      default: return <Info className="w-8 h-8 text-slate-500" />;
    }
  };

  return (
    <div className="w-full max-w-5xl mx-auto space-y-8">
      <div className="text-center space-y-3">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tighter text-white">
          Forensic Analysis Engine
        </h1>
        <p className="text-slate-400 font-mono text-sm uppercase tracking-widest">
          Advanced Multi-Vector Deepfake Detection
        </p>
      </div>

      <div className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 overflow-hidden shadow-2xl">
        <div className="flex border-b border-slate-800">
          <button
            onClick={() => { setActiveTab('media'); setResult(null); }}
            className={cn(
              "flex-1 py-4 text-sm font-mono uppercase tracking-wider transition-colors flex items-center justify-center gap-2",
              activeTab === 'media' ? "bg-indigo-500/10 text-indigo-400 border-b-2 border-indigo-500" : "text-slate-500 hover:bg-slate-800/50 hover:text-slate-300"
            )}
          >
            <ScanSearch className="w-4 h-4" />
            Media Scanner
          </button>
          <button
            onClick={() => { setActiveTab('text'); setResult(null); }}
            className={cn(
              "flex-1 py-4 text-sm font-mono uppercase tracking-wider transition-colors flex items-center justify-center gap-2",
              activeTab === 'text' ? "bg-indigo-500/10 text-indigo-400 border-b-2 border-indigo-500" : "text-slate-500 hover:bg-slate-800/50 hover:text-slate-300"
            )}
          >
            <FileText className="w-4 h-4" />
            Text Analysis
          </button>
        </div>

        <div className="p-6 md:p-8">
          {activeTab === 'media' ? (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2 space-y-6">
                {!file ? (
                  <div
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-slate-700 rounded-xl p-12 text-center cursor-pointer hover:bg-slate-800/50 hover:border-indigo-500/50 transition-all flex flex-col items-center justify-center min-h-[350px] group h-full"
                  >
                    <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                      <Upload className="w-8 h-8 text-indigo-400" />
                    </div>
                    <p className="text-slate-300 font-medium text-lg">Initialize Media Scan</p>
                    <p className="text-slate-500 text-sm mt-2 font-mono">Drag & drop or click to browse</p>
                    <p className="text-slate-600 text-xs mt-4 font-mono">JPG, PNG, MP4, WEBM, MP3, WAV (Max 15MB)</p>
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileChange}
                      accept="image/*,video/*,audio/*"
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-black min-h-[350px] flex items-center justify-center group h-full">
                    <button
                      onClick={clearFile}
                      className="absolute top-4 right-4 p-2 bg-slate-900/80 backdrop-blur-md rounded-lg hover:bg-rose-500/20 hover:text-rose-400 text-slate-400 border border-slate-700 transition-colors z-20"
                    >
                      <X className="w-4 h-4" />
                    </button>
                    
                    {isAnalyzing && (
                      <div className="absolute inset-0 z-10 pointer-events-none">
                        <div className="w-full h-full scanner-overlay opacity-70"></div>
                        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60 backdrop-blur-sm">
                          <div className="text-center space-y-4">
                            <Fingerprint className="w-12 h-12 text-indigo-400 animate-pulse mx-auto" />
                            <p className="text-indigo-300 font-mono text-sm animate-pulse">
                              {ANALYSIS_STEPS[analysisStep]}
                            </p>
                          </div>
                        </div>
                      </div>
                    )}

                    {file.type.startsWith('image/') ? (
                      <img src={URL.createObjectURL(file)} alt="Preview" className="max-h-[500px] object-contain" />
                    ) : file.type.startsWith('audio/') ? (
                      <div className="w-full p-8 flex flex-col items-center justify-center space-y-6">
                        <Mic className="w-20 h-20 text-indigo-500/50" />
                        <audio src={URL.createObjectURL(file)} controls className="w-full max-w-md" />
                      </div>
                    ) : (
                      <video src={URL.createObjectURL(file)} controls={!isAnalyzing} autoPlay loop muted className="max-h-[500px] w-full" />
                    )}
                  </div>
                )}
              </div>
              <div className="lg:col-span-1">
                <div className="bg-slate-950/50 rounded-xl border border-slate-800 p-4 h-full flex flex-col min-h-[350px]">
                  <h3 className="text-xs font-mono text-indigo-400 uppercase tracking-wider mb-4 flex items-center gap-2 border-b border-slate-800 pb-2">
                    <Database className="w-4 h-4" /> File Metadata
                  </h3>
                  {fileMeta ? (
                    <div className="space-y-4 flex-1 flex flex-col">
                      <div className="space-y-1">
                        <p className="text-xs text-slate-500 font-mono">FILENAME</p>
                        <p className="text-sm text-slate-300 truncate" title={fileMeta.name}>{fileMeta.name}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-slate-500 font-mono">SIZE</p>
                        <p className="text-sm text-slate-300">{fileMeta.size}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-xs text-slate-500 font-mono">MIME TYPE</p>
                        <p className="text-sm text-slate-300">{fileMeta.type}</p>
                      </div>
                      <div className="space-y-1 flex-1 flex flex-col">
                        <p className="text-xs text-slate-500 font-mono mb-1">HEX SIGNATURE (HEAD)</p>
                        <div className="bg-black border border-slate-800 rounded p-2 flex-1 overflow-y-auto max-h-[150px] md:max-h-none">
                          <pre className="text-[10px] text-emerald-500 font-mono whitespace-pre-wrap leading-tight">
                            {hexDump}
                          </pre>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex-1 flex items-center justify-center text-slate-600 text-xs font-mono text-center">
                      AWAITING FILE INPUT...
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-6">
              <div className="relative">
                {isAnalyzing && (
                  <div className="absolute inset-0 z-10 bg-slate-950/80 backdrop-blur-sm rounded-xl flex items-center justify-center border border-indigo-500/30">
                    <div className="text-center space-y-4">
                      <BrainCircuit className="w-12 h-12 text-indigo-400 animate-pulse mx-auto" />
                      <p className="text-indigo-300 font-mono text-sm animate-pulse">
                        {ANALYSIS_STEPS[analysisStep]}
                      </p>
                    </div>
                  </div>
                )}
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Input text sequence for semantic and structural analysis..."
                  className="w-full h-[350px] p-6 rounded-xl border border-slate-700 bg-slate-900/50 text-slate-300 focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500 resize-none font-mono text-sm leading-relaxed"
                />
              </div>
            </div>
          )}

          {error && (
            <div className="mt-6 p-4 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-xl flex items-start gap-3 text-sm font-mono">
              <AlertTriangle className="w-5 h-5 shrink-0" />
              <p>{error}</p>
            </div>
          )}

          <div className="mt-8 flex justify-end">
            <button
              onClick={analyze}
              disabled={isAnalyzing || (activeTab === 'media' ? !file : !text.trim())}
              className="px-8 py-4 bg-indigo-600 text-white font-mono text-sm uppercase tracking-wider rounded-xl hover:bg-indigo-500 focus:ring-4 focus:ring-indigo-500/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3 transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)]"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  PROCESSING...
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5" />
                  EXECUTE ANALYSIS
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          >
            {/* Verdict Card */}
            <div className={cn(
              "lg:col-span-1 rounded-2xl p-6 border flex flex-col items-center justify-center text-center space-y-4 relative overflow-hidden",
              getVerdictColor(result.verdict)
            )}>
              <button 
                onClick={downloadReport}
                className="absolute top-4 right-4 p-2 bg-slate-900/50 hover:bg-slate-800 rounded-lg transition-colors border border-slate-700 text-slate-300"
                title="Download Forensic Report"
              >
                <Download className="w-4 h-4" />
              </button>
              {getVerdictIcon(result.verdict)}
              <div>
                <h2 className="text-3xl font-bold tracking-tight mb-1">
                  {result.verdict}
                </h2>
                <p className="font-mono text-sm opacity-80 uppercase tracking-widest">
                  Confidence: {result.confidenceScore}%
                </p>
              </div>
            </div>

            {/* Summary & Indicators */}
            <div className="lg:col-span-2 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 p-6 space-y-6">
              <div>
                <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <BrainCircuit className="w-4 h-4" /> Executive Summary
                </h3>
                <p className="text-slate-300 leading-relaxed">
                  {result.executiveSummary}
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {result.riskFactors.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-mono text-rose-400 uppercase tracking-wider flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" /> Risk Factors
                    </h3>
                    <ul className="space-y-2">
                      {result.riskFactors.map((factor, idx) => (
                        <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                          <span className="text-rose-500 mt-1">â–¸</span> {factor}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {result.authenticityIndicators.length > 0 && (
                  <div className="space-y-3">
                    <h3 className="text-sm font-mono text-emerald-400 uppercase tracking-wider flex items-center gap-2">
                      <ShieldCheck className="w-4 h-4" /> Authenticity Indicators
                    </h3>
                    <ul className="space-y-2">
                      {result.authenticityIndicators.map((indicator, idx) => (
                        <li key={idx} className="text-sm text-slate-400 flex items-start gap-2">
                          <span className="text-emerald-500 mt-1">â–¸</span> {indicator}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>

            {/* Detailed Forensic Breakdown */}
            <div className="lg:col-span-3 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 p-6">
              <h3 className="text-sm font-mono text-slate-400 uppercase tracking-wider mb-6 flex items-center gap-2 border-b border-slate-800 pb-4">
                <Radar className="w-4 h-4" /> Anomaly Vector Mapping
              </h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-1 h-[300px] bg-slate-950/50 rounded-xl border border-slate-800/50 flex items-center justify-center p-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={[
                      { subject: 'Visual', A: result.vectorScores.visual, fullMark: 100 },
                      { subject: 'Audio', A: result.vectorScores.audio, fullMark: 100 },
                      { subject: 'Temporal', A: result.vectorScores.temporal, fullMark: 100 },
                      { subject: 'Logical', A: result.vectorScores.logical, fullMark: 100 },
                    ]}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'monospace' }} />
                      <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                      <RechartsRadar name="Anomaly Score" dataKey="A" stroke="#6366f1" fill="#6366f1" fillOpacity={0.4} />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50">
                    <h4 className="text-xs font-mono text-indigo-400 uppercase tracking-wider mb-2 flex justify-between">
                      <span>Visual / Structural</span>
                      <span className="text-slate-500">{result.vectorScores.visual}/100</span>
                    </h4>
                    <p className="text-sm text-slate-400">{result.forensicAnalysis.visualArtifacts}</p>
                  </div>
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50">
                    <h4 className="text-xs font-mono text-indigo-400 uppercase tracking-wider mb-2 flex justify-between">
                      <span>Audio / Semantic</span>
                      <span className="text-slate-500">{result.vectorScores.audio}/100</span>
                    </h4>
                    <p className="text-sm text-slate-400">{result.forensicAnalysis.lightingAndShadows}</p>
                  </div>
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50">
                    <h4 className="text-xs font-mono text-indigo-400 uppercase tracking-wider mb-2 flex justify-between">
                      <span>Anatomy & Logic</span>
                      <span className="text-slate-500">{result.vectorScores.logical}/100</span>
                    </h4>
                    <p className="text-sm text-slate-400">{result.forensicAnalysis.anatomyAndPhysics}</p>
                  </div>
                  <div className="bg-slate-950/50 rounded-xl p-4 border border-slate-800/50">
                    <h4 className="text-xs font-mono text-indigo-400 uppercase tracking-wider mb-2 flex justify-between">
                      <span>Temporal Consistency</span>
                      <span className="text-slate-500">{result.vectorScores.temporal}/100</span>
                    </h4>
                    <p className="text-sm text-slate-400">{result.forensicAnalysis.temporalConsistency}</p>
                  </div>
                </div>
              </div>
            </div>

          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
