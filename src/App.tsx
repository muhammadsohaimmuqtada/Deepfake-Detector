/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { DeepfakeDetector } from './components/DeepfakeDetector';
import { Chatbot } from './components/Chatbot';
import { ShieldAlert, Key, Terminal, Lock } from 'lucide-react';

declare global {
  interface Window {
    aistudio?: {
      hasSelectedApiKey: () => Promise<boolean>;
      openSelectKey: () => Promise<void>;
    };
  }
}

export default function App() {
  const [isAuthorized, setIsAuthorized] = useState(false);
  const [isChecking, setIsChecking] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        if (window.aistudio && window.aistudio.hasSelectedApiKey) {
          const hasKey = await window.aistudio.hasSelectedApiKey();
          setIsAuthorized(hasKey);
        } else {
          setIsAuthorized(true);
        }
      } catch (e) {
        setIsAuthorized(true);
      } finally {
        setIsChecking(false);
      }
    };
    checkAuth();
  }, []);

  const handleAuthenticate = async () => {
    if (window.aistudio && window.aistudio.openSelectKey) {
      await window.aistudio.openSelectKey();
      setIsAuthorized(true);
    } else {
      setIsAuthorized(true);
    }
  };

  if (isChecking) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center text-indigo-400 font-mono grid-bg">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
          <p className="tracking-widest animate-pulse">INITIALIZING CORE SYSTEMS...</p>
        </div>
      </div>
    );
  }

  if (!isAuthorized) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4 font-mono grid-bg">
        <div className="max-w-md w-full bg-slate-900/80 backdrop-blur-xl border border-rose-500/30 p-8 rounded-2xl shadow-[0_0_30px_rgba(244,63,94,0.1)] text-center space-y-6">
          <div className="w-20 h-20 bg-rose-500/10 border border-rose-500/30 rounded-full flex items-center justify-center mx-auto">
            <Lock className="w-10 h-10 text-rose-500" />
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl font-bold text-white tracking-widest">SYSTEM LOCKED</h1>
            <p className="text-slate-400 text-sm leading-relaxed">
              DeepfakeGUARD requires a dedicated Google Cloud API Key to perform advanced forensic analysis. 
              Please authenticate to initialize the neural pathways.
            </p>
          </div>
          
          <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-left space-y-3 text-xs text-slate-500">
            <p className="flex items-center gap-3"><Terminal className="w-4 h-4 text-indigo-500" /> Connects to Gemini 3.1 Pro</p>
            <p className="flex items-center gap-3"><Terminal className="w-4 h-4 text-indigo-500" /> Enables Multi-Vector Analysis</p>
            <p className="flex items-center gap-3"><Terminal className="w-4 h-4 text-indigo-500" /> Secures Forensic Comms</p>
          </div>

          <button
            onClick={handleAuthenticate}
            className="w-full py-4 bg-indigo-600 hover:bg-indigo-500 text-white font-bold tracking-widest rounded-xl transition-all shadow-[0_0_20px_rgba(99,102,241,0.3)] hover:shadow-[0_0_30px_rgba(99,102,241,0.5)] flex items-center justify-center gap-3"
          >
            <Key className="w-5 h-5" />
            AUTHENTICATE NOW
          </button>
          
          <p className="text-xs text-slate-600 mt-4">
            <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noreferrer" className="underline hover:text-indigo-400">
              View Billing Documentation
            </a>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 font-sans text-slate-50 selection:bg-indigo-500/30 selection:text-indigo-200 grid-bg">
      <header className="bg-slate-950/80 backdrop-blur-md border-b border-slate-800 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-indigo-500/10 border border-indigo-500/30 rounded-xl flex items-center justify-center shadow-[0_0_15px_rgba(99,102,241,0.2)]">
              <ShieldAlert className="w-5 h-5 text-indigo-400" />
            </div>
            <span className="font-bold text-xl tracking-tight text-white">
              Deepfake<span className="text-indigo-400 font-mono ml-1">GUARD</span>
            </span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm font-mono text-slate-400">
            <a href="#" className="hover:text-indigo-400 transition-colors uppercase tracking-wider text-xs">Analysis Engine</a>
            <a href="#" className="hover:text-indigo-400 transition-colors uppercase tracking-wider text-xs">Forensic DB</a>
            <a href="#" className="hover:text-indigo-400 transition-colors uppercase tracking-wider text-xs flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div> API Connected</a>
          </nav>
        </div>
      </header>

      <main className="py-12 px-4 sm:px-6 lg:px-8 relative z-10">
        <DeepfakeDetector />
      </main>

      <Chatbot />
    </div>
  );
}
