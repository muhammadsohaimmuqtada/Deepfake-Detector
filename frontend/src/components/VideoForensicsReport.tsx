/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  Mic,
  Heart,
  Eye,
  Download,
  Clock,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { motion } from 'motion/react';
import { cn } from '@/lib/utils';
import type { ForensicReport } from '@/lib/analyzeService';

interface Props {
  report: ForensicReport;
  filename: string;
  onReset: () => void;
}

/** Formats a 0-1 probability as a rounded percentage string. */
function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Score bar — fills left-to-right, coloured by risk level. */
function ScoreBar({ value, isBad }: { value: number; isBad: boolean }) {
  const fill = Math.min(Math.max(value * 100, 0), 100);
  return (
    <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden mt-2">
      <div
        className={cn(
          'h-full rounded-full transition-all duration-700',
          isBad ? 'bg-rose-500' : 'bg-emerald-500',
        )}
        style={{ width: `${fill}%` }}
      />
    </div>
  );
}

/** Individual forensic pillar card. */
function PillarCard({
  icon,
  title,
  subtitle,
  isFake,
  score,
  confidence,
  details,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  isFake: boolean;
  score: number;
  confidence: number;
  details: { label: string; value: string }[];
}) {
  const statusLabel = isFake ? 'THREAT DETECTED' : 'PASS';
  const borderColor = isFake
    ? 'border-rose-500/30 shadow-[0_0_20px_rgba(244,63,94,0.08)]'
    : 'border-emerald-500/30 shadow-[0_0_20px_rgba(16,185,129,0.08)]';
  const accentColor = isFake ? 'text-rose-400' : 'text-emerald-400';
  const bgAccent = isFake ? 'bg-rose-500/10' : 'bg-emerald-500/10';

  return (
    <div
      className={cn(
        'rounded-2xl border bg-slate-900/60 backdrop-blur-xl p-6 flex flex-col gap-4',
        borderColor,
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between">
        <div
          className={cn(
            'w-10 h-10 rounded-xl flex items-center justify-center',
            bgAccent,
          )}
        >
          <span className={accentColor}>{icon}</span>
        </div>
        <span
          className={cn(
            'text-xs font-mono font-bold tracking-widest px-2 py-1 rounded-lg border',
            isFake
              ? 'text-rose-400 border-rose-500/30 bg-rose-500/10'
              : 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
          )}
        >
          {statusLabel}
        </span>
      </div>

      {/* Title */}
      <div>
        <h3 className="text-white font-semibold tracking-tight">{title}</h3>
        <p className="text-slate-500 text-xs font-mono mt-0.5">{subtitle}</p>
      </div>

      {/* Confidence */}
      <div>
        <div className="flex justify-between items-center text-xs font-mono">
          <span className="text-slate-500 uppercase tracking-wider">
            Detection Score
          </span>
          <span className={cn('font-bold', accentColor)}>
            {Math.round(score * 100)}%
          </span>
        </div>
        <ScoreBar value={score} isBad={isFake} />
      </div>

      {/* Detail rows */}
      <div className="space-y-2 border-t border-slate-800 pt-4">
        {details.map((d) => (
          <div
            key={d.label}
            className="flex justify-between items-center text-xs font-mono"
          >
            <span className="text-slate-500">{d.label}</span>
            <span className="text-slate-300">{d.value}</span>
          </div>
        ))}
        <div className="flex justify-between items-center text-xs font-mono">
          <span className="text-slate-500">Confidence</span>
          <span className="text-slate-300">{Math.round(confidence)}%</span>
        </div>
      </div>
    </div>
  );
}

export function VideoForensicsReport({ report, filename, onReset }: Props) {
  const isFake = report.is_fake;
  const verdictLabel = isFake ? 'SYNTHETIC THREAT DETECTED' : 'AUTHENTIC';

  // ── Layer 1 — Visual / FFT ──────────────────────────────────────────────────
  const fftScore = report.fake_frame_ratio ?? 0;
  const fftIsFake = fftScore >= 0.4 || (report.frames_flagged ?? 0) > 0 && fftScore > 0.2;
  const fftConfidence =
    report.layer_weights != null
      ? (report.layer_weights.visual_fft ?? 0) * 100
      : 50;

  // ── Layer 2 — Biological / rPPG ───────────────────────────────────────────
  const rppg = report.layer2_rppg_report;
  const rppgIsFake = rppg?.is_fake ?? false;
  const rppgScore = rppg != null ? (rppg.is_fake ? rppg.confidence / 100 : 1 - rppg.confidence / 100) : 0.5;
  const rppgConfidence = rppg?.confidence ?? 0;

  // ── Layer 3 — Audio ────────────────────────────────────────────────────────
  const audio = report.layer3_audio_report;
  const audioIsFake = audio?.is_fake ?? false;
  const audioScore = audio != null ? (audio.is_fake ? audio.confidence / 100 : 1 - audio.confidence / 100) : 0.5;
  const audioConfidence = audio?.confidence ?? 0;

  // ── Download handler ───────────────────────────────────────────────────────
  const handleDownload = () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_report_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -40 }}
      className="space-y-6"
    >
      {/* ── Overall Verdict Banner ─────────────────────────────────────────── */}
      <div
        className={cn(
          'rounded-2xl border p-8 flex flex-col md:flex-row items-center justify-between gap-6 relative overflow-hidden',
          isFake
            ? 'bg-rose-950/30 border-rose-500/40 shadow-[0_0_40px_rgba(244,63,94,0.12)]'
            : 'bg-emerald-950/30 border-emerald-500/40 shadow-[0_0_40px_rgba(16,185,129,0.12)]',
        )}
      >
        {/* Background glow */}
        <div
          className={cn(
            'absolute inset-0 pointer-events-none',
            isFake
              ? 'bg-gradient-to-br from-rose-900/20 to-transparent'
              : 'bg-gradient-to-br from-emerald-900/20 to-transparent',
          )}
        />

        <div className="flex items-center gap-5 relative z-10">
          <div
            className={cn(
              'w-16 h-16 rounded-2xl flex items-center justify-center border',
              isFake
                ? 'bg-rose-500/10 border-rose-500/30'
                : 'bg-emerald-500/10 border-emerald-500/30',
            )}
          >
            {isFake ? (
              <ShieldAlert className="w-8 h-8 text-rose-400" />
            ) : (
              <ShieldCheck className="w-8 h-8 text-emerald-400" />
            )}
          </div>

          <div>
            <p className="text-xs font-mono text-slate-400 uppercase tracking-widest mb-1">
              Final Verdict
            </p>
            <h2
              className={cn(
                'text-2xl md:text-3xl font-bold tracking-tight',
                isFake ? 'text-rose-400' : 'text-emerald-400',
              )}
            >
              {verdictLabel}
            </h2>
            <p className="text-slate-400 text-sm font-mono mt-1">
              {filename}
            </p>
          </div>
        </div>

        <div className="flex flex-col items-center gap-1 relative z-10">
          <p className="text-xs font-mono text-slate-500 uppercase tracking-widest">
            Confidence Score
          </p>
          <p
            className={cn(
              'text-5xl font-bold font-mono',
              isFake ? 'text-rose-400' : 'text-emerald-400',
            )}
          >
            {report.overall_confidence}
            <span className="text-2xl">%</span>
          </p>
          <div className="flex items-center gap-1.5 mt-1">
            <Clock className="w-3.5 h-3.5 text-slate-500" />
            <span className="text-xs font-mono text-slate-500">
              Analysis time: {report.elapsed_seconds}s
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 relative z-10">
          <button
            onClick={handleDownload}
            title="Download full JSON report"
            className="p-3 bg-slate-900/80 border border-slate-700 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
          >
            <Download className="w-5 h-5" />
          </button>
          <button
            onClick={onReset}
            title="Analyze another video"
            className="px-4 py-2 bg-slate-900/80 border border-slate-700 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-white transition-colors text-xs font-mono uppercase tracking-wider"
          >
            New Scan
          </button>
        </div>
      </div>

      {/* ── Inconclusive warning ───────────────────────────────────────────── */}
      {report.verdict === 'INCONCLUSIVE' && (
        <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 text-amber-400 text-sm font-mono">
          <AlertTriangle className="w-5 h-5 shrink-0 mt-0.5" />
          <p>
            {report.message ??
              'Analysis was inconclusive. No faces could be detected. The video may contain no human subjects or may be too low quality.'}
          </p>
        </div>
      )}

      {/* ── 3-Pillar Cards ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Card 1 — Visual Artifacts (FFT) */}
        <PillarCard
          icon={<Eye className="w-5 h-5" />}
          title="Visual Artifacts"
          subtitle="Layer 1 · FFT Spectral Analysis"
          isFake={fftIsFake}
          score={fftScore}
          confidence={fftConfidence}
          details={[
            {
              label: 'Frames Analyzed',
              value: String(report.frames_analyzed ?? 0),
            },
            {
              label: 'Frames Flagged',
              value: String(report.frames_flagged ?? 0),
            },
            {
              label: 'Flag Ratio',
              value: pct(report.fake_frame_ratio ?? 0),
            },
            {
              label: 'Artifact Score',
              value: (report.mean_artifact_score ?? 0).toFixed(4),
            },
          ]}
        />

        {/* Card 2 — Biological Check (rPPG) */}
        <PillarCard
          icon={<Heart className="w-5 h-5" />}
          title="Biological Pulse"
          subtitle="Layer 2 · rPPG Heartbeat Analysis"
          isFake={rppgIsFake}
          score={rppgScore}
          confidence={rppgConfidence}
          details={
            rppg != null && rppg.verdict !== 'INCONCLUSIVE'
              ? [
                  {
                    label: 'rPPG SNR',
                    value: rppg.rppg_snr?.toFixed(3) ?? '—',
                  },
                  {
                    label: 'Dominant BPM',
                    value: rppg.dominant_bpm?.toFixed(1) ?? '—',
                  },
                  {
                    label: 'Frames Used',
                    value: String(rppg.frames_analyzed ?? 0),
                  },
                ]
              : [
                  {
                    label: 'Status',
                    value: rppg?.reason ?? 'INCONCLUSIVE',
                  },
                ]
          }
        />

        {/* Card 3 — Audio Forensics */}
        <PillarCard
          icon={<Mic className="w-5 h-5" />}
          title="Audio Forensics"
          subtitle="Layer 3 · Acoustic / MFCC Analysis"
          isFake={audioIsFake}
          score={audioScore}
          confidence={audioConfidence}
          details={
            audio != null && audio.verdict !== 'INCONCLUSIVE'
              ? [
                  {
                    label: 'Artifact Score',
                    value: audio.artifact_score?.toFixed(4) ?? '—',
                  },
                  {
                    label: 'Duration',
                    value:
                      audio.audio_duration_seconds != null
                        ? `${audio.audio_duration_seconds.toFixed(1)}s`
                        : '—',
                  },
                  {
                    label: 'Spectral Flatness',
                    value: audio.spectral_flatness_mean?.toFixed(5) ?? '—',
                  },
                ]
              : [
                  {
                    label: 'Status',
                    value: audio?.reason ?? 'INCONCLUSIVE',
                  },
                ]
          }
        />
      </div>

      {/* ── Ensemble Summary ──────────────────────────────────────────────── */}
      {report.ensemble_fake_probability != null && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
          <h3 className="text-xs font-mono text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            Ensemble Scoring Matrix
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {report.layer_weights && (
              <>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-500">Visual / FFT Weight</span>
                    <span className="text-indigo-300">
                      {pct(report.layer_weights.visual_fft)}
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-indigo-500 rounded-full"
                      style={{
                        width: pct(report.layer_weights.visual_fft),
                      }}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-500">Biological / rPPG Weight</span>
                    <span className="text-violet-300">
                      {pct(report.layer_weights.biological_rppg)}
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-violet-500 rounded-full"
                      style={{
                        width: pct(report.layer_weights.biological_rppg),
                      }}
                    />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-500">Acoustic / Audio Weight</span>
                    <span className="text-cyan-300">
                      {pct(report.layer_weights.acoustic_audio)}
                    </span>
                  </div>
                  <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                    <div
                      className="h-full bg-cyan-500 rounded-full"
                      style={{
                        width: pct(report.layer_weights.acoustic_audio),
                      }}
                    />
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="mt-4 pt-4 border-t border-slate-800 flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-2">
              {isFake ? (
                <AlertTriangle className="w-4 h-4 text-rose-400" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              )}
              <span className="text-slate-400">
                Ensemble fake probability:{' '}
                <span
                  className={isFake ? 'text-rose-400' : 'text-emerald-400'}
                >
                  {pct(report.ensemble_fake_probability)}
                </span>
              </span>
            </div>
            <span className="text-slate-600">
              Decision threshold: 50%
            </span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
