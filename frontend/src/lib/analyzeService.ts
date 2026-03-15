/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * analyzeService — Sends a video file to the FastAPI deepfake detection backend
 * and returns the structured forensic report.
 */

const BACKEND_URL = 'http://localhost:8000';

export interface RPPGLayerReport {
  module: string;
  is_fake: boolean;
  confidence: number;
  rppg_snr: number;
  dominant_frequency_hz: number;
  dominant_bpm: number;
  frames_analyzed: number;
  snr_threshold: number;
  verdict?: string;
  reason?: string;
}

export interface AudioLayerReport {
  module: string;
  is_fake: boolean;
  artifact_score: number;
  confidence: number;
  mfcc_mean_variance: number;
  mfcc_variance_threshold: number;
  mfcc_delta_mean_variance: number;
  mfcc_delta2_mean_variance: number;
  spectral_flatness_mean: number;
  spectral_flatness_threshold: number;
  zcr_mean: number;
  zcr_variance: number;
  audio_duration_seconds: number;
  sample_rate: number;
  verdict?: string;
  reason?: string;
}

export interface ForensicReport {
  video_path: string;
  verdict: 'FAKE' | 'REAL' | 'INCONCLUSIVE';
  is_fake: boolean;
  overall_confidence: number;
  frames_analyzed: number;
  frames_flagged: number;
  fake_frame_ratio: number;
  mean_artifact_score: number;
  ensemble_fake_probability?: number;
  layer_weights?: {
    visual_fft: number;
    biological_rppg: number;
    acoustic_audio: number;
  };
  elapsed_seconds: number;
  layer1_fft_report: Record<string, unknown>[];
  layer2_rppg_report: RPPGLayerReport | null;
  layer3_audio_report: AudioLayerReport;
  message?: string;
}

/**
 * Posts an .mp4 video file to the FastAPI backend for forensic analysis.
 *
 * Throws a descriptive Error if:
 *  - The backend is unreachable (connection refused / network error).
 *  - The backend returns a non-2xx HTTP status.
 */
export async function analyzeVideo(file: File): Promise<ForensicReport> {
  const formData = new FormData();
  formData.append('file', file);

  let response: Response;
  try {
    response = await fetch(`${BACKEND_URL}/api/analyze`, {
      method: 'POST',
      body: formData,
    });
  } catch {
    throw new Error(
      'Cannot connect to the Forensic Engine. Please ensure the FastAPI backend is running on http://localhost:8000 (run: uvicorn main:app --reload).',
    );
  }

  if (!response.ok) {
    const errorData = await response
      .json()
      .catch(() => ({ detail: `HTTP ${response.status}` }));
    throw new Error(
      errorData.detail || `Analysis engine returned error: ${response.status}`,
    );
  }

  return response.json() as Promise<ForensicReport>;
}
