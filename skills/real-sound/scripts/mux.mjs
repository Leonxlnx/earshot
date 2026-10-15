#!/usr/bin/env node
// Put a soundtrack on the finished picture without re-encoding the video (stream copy; AAC 48 kHz stereo audio).
//   node <skill>/scripts/mux.mjs --video out/film.mp4 [--audio out/audio/soundtrack.wav] [--out out/film_sound.mp4]
//                                [--abr 320k] [--keep-old out/superseded/film_prev.mp4] [--root .]
// Relative paths resolve against --root (default: the current directory). The audio is cut to the video's own duration
// (ffprobe), so a soundtrack that runs long never extends the film; a length mismatch over one frame is reported.
// When --out is the --video file itself, the old file is moved to --keep-old first (default
// out/superseded/<name>_prev<ext>); an existing keep-old file is never overwritten. Exit 0 = written and verified.
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';

const argv = process.argv.slice(2);
if (argv.includes('-h') || argv.includes('--help')) {
  console.log(fs.readFileSync(new URL(import.meta.url), 'utf8').split('\n').slice(1, 8).map((l) => l.replace(/^\/\/ ?/, '')).join('\n'));
  process.exit(0);
}
const opt = (k, d) => { const i = argv.indexOf(`--${k}`); return i >= 0 && argv[i + 1] ? argv[i + 1] : d; };
const ROOT = path.resolve(opt('root', '.'));
const R = (p) => (path.isAbsolute(p) ? p : path.join(ROOT, p));
const rel = (p) => path.relative(ROOT, p) || p;
const fail = (m) => { console.error(`mux: ${m}`); process.exit(1); };

function probe(file) {
  const r = spawnSync('ffprobe', ['-v', 'error', '-show_entries', 'stream=codec_type,codec_name,r_frame_rate,duration:format=duration',
    '-of', 'json', file], { encoding: 'utf8' });
  if (r.status !== 0) fail(`ffprobe failed on ${file}: ${(r.stderr || '').slice(0, 300)}`);
  const j = JSON.parse(r.stdout || '{}');
  const v = (j.streams || []).find((s) => s.codec_type === 'video');
  const a = (j.streams || []).find((s) => s.codec_type === 'audio');
  const [n, d] = ((v && v.r_frame_rate) || '0/1').split('/').map(Number);
  const fmt = Number((j.format || {}).duration || 0);
  return { video: v, audio: a, fps: d ? n / d : 0, duration: Number((v && v.duration) || fmt), audioDuration: Number((a && a.duration) || fmt) };
}

const audio = R(opt('audio', 'out/audio/soundtrack.wav'));
let video = R(opt('video', 'out/film.mp4'));
const out = R(opt('out', video));
const abr = opt('abr', '320k');
if (!fs.existsSync(audio)) fail(`no audio ${audio} (pass --audio)`);
if (!fs.existsSync(video)) fail(`no video ${video} (pass --video)`);
const pv = probe(video);
if (!pv.video) fail(`${video} has no video stream`);
const pa = probe(audio);
const frame = pv.fps ? 1 / pv.fps : 1 / 60;
const diff = pa.audioDuration - pv.duration;
if (Math.abs(diff) > frame) {
  console.log(`note: the audio is ${diff > 0 ? 'longer' : 'SHORTER'} than the picture by ${Math.abs(diff).toFixed(3)} s ` +
    `(${(Math.abs(diff) / frame).toFixed(1)} frames)` + (diff > 0 ? ': it is cut at the picture\'s end' : ': the end will be silent'));
}
if (path.resolve(out) === path.resolve(video)) {
  const ext = path.extname(video);
  const keep = R(opt('keep-old', path.join('out', 'superseded', path.basename(video, ext) + '_prev' + ext)));
  fs.mkdirSync(path.dirname(keep), { recursive: true });
  if (fs.existsSync(keep)) fail(`refusing to overwrite ${keep}; pass --keep-old <new path>`);
  fs.renameSync(video, keep);
  video = keep;
  console.log(`previous version kept at ${rel(keep)}`);
}
fs.mkdirSync(path.dirname(out), { recursive: true });
const ext = path.extname(out) || '.mp4';
const tmp = out.slice(0, out.length - path.extname(out).length) + '.muxing' + ext;
const args = ['-v', 'error', '-y', '-i', video, '-i', audio, '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
  '-b:a', abr, '-ar', '48000', '-ac', '2', '-t', pv.duration.toFixed(6)];
if (/\.(mp4|m4v|mov)$/i.test(ext)) args.push('-movflags', '+faststart');
const r = spawnSync('ffmpeg', [...args, tmp], { stdio: 'inherit' });
if (r.status !== 0) { try { fs.unlinkSync(tmp); } catch { /* nothing to clean */ } fail('ffmpeg failed'); }
const po = probe(tmp);
if (!po.video || !po.audio) fail(`${tmp} lacks a video or audio stream`);
fs.renameSync(tmp, out);
console.log(`wrote ${rel(out)} (${(fs.statSync(out).size / 1e6).toFixed(1)} MB): video copied (${pv.video.codec_name}, ` +
  `${pv.fps.toFixed(3)} fps, ${pv.duration.toFixed(3)} s), audio ${po.audio.codec_name} ${abr} ${po.audioDuration.toFixed(3)} s`);
