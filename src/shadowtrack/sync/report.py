"""Standalone HTML and Matplotlib PNG figures; no decoding during report creation."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..probe import AuditError
from .api import TimeAlignment
from .fit import predict_interval


def render(alignment_path: Path, output: Path | None = None):
    alignment_path = alignment_path.resolve()
    api = TimeAlignment.load(alignment_path)
    doc = api.document
    output = (output or alignment_path.parent).resolve()
    if output.is_relative_to(Path(doc["source_stats"]).parent) or output.is_relative_to(Path.cwd()/"data"):
        raise AuditError("Report output must be separate from source data and the completed audit")
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    count = len(doc["cameras"])
    fig,axes = plt.subplots(count,2,figsize=(14,max(3,2.9*count)),squeeze=False,layout="constrained")
    for row,camera in enumerate(doc["cameras"]):
        timeline = api.timelines[camera["camera_id"]]
        frame_index = np.arange(len(timeline.raw))
        raw_relative = timeline.raw-timeline.raw[np.flatnonzero(np.isfinite(timeline.raw))[0]]
        ax,delta_ax = axes[row]
        ax.plot(frame_index,raw_relative,label="raw PTS - first PTS",color="#687782",lw=1.4)
        ax.plot(frame_index,timeline.estimate,label="local estimate (not global)",color="#087c91",lw=1.1)
        ax.fill_between(frame_index,timeline.lower,timeline.upper,color="#087c91",alpha=.13,label="local correction bounds")
        for anomaly in timeline.anomalies:
            ax.axvline(anomaly["frame_after"],color="#a45b16" if anomaly["corrected"] else "#af4040",alpha=.65,lw=.8)
        ax.set(title=f"Camera {camera['camera_id']} / {camera['synchronization_status']}",xlabel="Decoded frame_index (zero-based)",ylabel="Relative timeline (s)")
        correction_values = np.round(timeline.estimate-raw_relative,9)
        for segment in timeline.segments:
            lo,hi = segment["frame_start"],segment["frame_end"]+1
            delta_ax.plot(timeline.raw[lo:hi],correction_values[lo:hi],color="#087c91")
        if np.nanmax(np.abs(correction_values)) < 1e-8:
            delta_ax.set_ylim(-.1,.1)
        for anomaly in timeline.anomalies:
            if "after_pts" in anomaly:
                delta_ax.axvspan(anomaly["before_pts"],anomaly["after_pts"],color="#a45b16" if anomaly["corrected"] else "#af4040",alpha=.18)
        for event in doc["events"]:
            bounds = event.get("observations",{}).get(camera["camera_id"])
            if bounds and event.get("accepted_for_fit"):
                delta_ax.scatter([sum(bounds)/2],[0],marker="|",s=100,color="#20252a")
        for region in camera["supported_regions"]:
            a,b = region["raw_interval_candidate"]
            delta_ax.axvspan(a,b,color="#14835c",alpha=.3)
        delta_ax.set(xlabel="raw_media_pts (s)",ylabel="Correction to raw relative (s)",title="Local correction / red: unresolved / amber: corrected / green: global")
        delta_ax.title.set_fontsize(9)
        if row == 0:
            ax.legend(fontsize=8,loc="upper left")
    fig.suptitle("ShadowTrack: raw vs locally corrected timelines — all source MKV timestamps unchanged",fontsize=14)
    fig.savefig(output/"timelines.png",dpi=150)
    plt.close(fig)
    if doc["reference_support"]:
        fig,ax = plt.subplots(figsize=(10,3.5),layout="constrained")
        max_support = max(r["raw_interval_candidate"][1] for c in doc["cameras"] for r in c["supported_regions"])
        xmax = max_support+1
        for i,camera in enumerate(doc["cameras"]):
            timeline = api.timelines[camera["camera_id"]]
            start = float(timeline.raw[np.flatnonzero(np.isfinite(timeline.raw))[0]])
            ax.plot([start,xmax],[i,i],color="#c4cbd0",lw=10,solid_capstyle="butt")
            for region in camera["supported_regions"]:
                a,b = region["raw_interval_candidate"]
                ax.plot([a,b],[i,i],color="#14835c",lw=10,solid_capstyle="butt")
            if not camera["supported_regions"]:
                ax.text(.05,i+.18,"global unavailable",fontsize=8,color="#7b3830")
        ax.set(yticks=range(count),yticklabels=[f"Camera {c['camera_id']}" for c in doc["cameras"]],
               xlabel="raw_media_pts per camera (s)",ylabel="Camera ID",xlim=(0,xmax),
               title="Global calibration support (green); remaining raw time unsupported (gray)")
        ax.invert_yaxis()
        fig.savefig(output/"global_support.png",dpi=150)
        plt.close(fig)
    models = [m for m in doc["pair_models"] if m["status"] == "SUPPORTED"]
    if models:
        fig,axes = plt.subplots(len(models),2,figsize=(13,max(3,3.3*len(models))),squeeze=False,layout="constrained")
        for row,model in enumerate(models):
            raw = model["raw_pts_model"]
            if raw["status"] != "SUPPORTED":
                continue
            anchors = model["anchors"]
            xs = np.array([sum(a["raw_x"])/2 for a in anchors])
            ys = np.array([sum(a["raw_y"])/2 for a in anchors])
            xerr = np.array([(a["raw_x"][1]-a["raw_x"][0])/2 for a in anchors])
            yerr = np.array([(a["raw_y"][1]-a["raw_y"][0])/2 for a in anchors])
            ax,res = axes[row]
            ax.errorbar(xs,ys-xs,xerr=xerr,yerr=xerr+yerr,fmt="o",color="#087c91",capsize=3,label="checked TV event bounds")
            grid = np.linspace(*raw["x_support"],100)
            bounds = np.array([predict_interval(raw,float(x)) for x in grid])
            ax.plot(grid,(raw["slope"]-1)*grid+raw["intercept_seconds"],color="#18252c",label=raw["model"])
            ax.fill_between(grid,bounds[:,0]-grid,bounds[:,1]-grid,color="#087c91",alpha=.15,label="admissible parameter envelope")
            ax.set(title=model["id"]+" (pair diagnostic; not necessarily global)",xlabel="Camera A raw PTS (s)",ylabel="PTS B - PTS A (s)")
            ax.legend(fontsize=8)
            residual = ys-(raw["slope"]*xs+raw["intercept_seconds"])
            res.errorbar(xs,residual,yerr=xerr*raw["slope"]+yerr+raw["systematic_seconds"],fmt="o",color="#087c91",capsize=3)
            res.axhline(0,color="#687782",lw=.8)
            res.set(title=f"Residuals / center RMS {raw['rms_center_residual_seconds']:.4f} s",xlabel="Camera A raw PTS (s)",ylabel="Observed center - fitted PTS B (s)")
        fig.savefig(output/"anchors_residuals.png",dpi=150)
        plt.close(fig)
    audio = doc["audio"].get("cameras",{})
    if audio:
        fig,ax = plt.subplots(figsize=(9,3.4),layout="constrained")
        labels = list(audio)
        values = [audio[c]["metrics"]["ac_rms_dbfs"] for c in labels]
        ax.bar(labels,values,color="#687782")
        ax.axhline(-60,color="#af4040",label="energy gate (-60 dBFS)")
        ax.set(xlabel="Camera ID",ylabel="AC RMS level (dBFS)",title="Audio informativeness — energy alone does not prove common events",ylim=(-100,0))
        ax.legend()
        fig.savefig(output/"audio_quality.png",dpi=150)
        plt.close(fig)
    def e(v):
        return escape(str(v))
    camera_rows = []
    for c in doc["cameras"]:
        raw_domains = [r["raw_interval_candidate"] for r in c["supported_regions"]]
        camera_rows.append("<tr>"+"".join(f"<td>{e(v)}</td>" for v in [c["camera_id"],c["nominal_fps"],c["frame_count"],
                           sum(a["corrected"] for a in c["anomalies"]),c["synchronization_status"],raw_domains or "unavailable"])+"</tr>")
    pair_rows = []
    for m in doc["pair_models"]:
        r = m.get("raw_pts_model",{})
        pair_rows.append("<tr>"+"".join(f"<td>{e(v)}</td>" for v in [m["id"],m["status"],len(m["anchors"]),r.get("model"),
                         r.get("intercept_seconds"),r.get("drift_ppm"),r.get("x_support"),r.get("rms_center_residual_seconds")])+"</tr>")
    audio_rows = []
    for cid,c in audio.items():
        q = c["metrics"]
        audio_rows.append("<tr>"+"".join(f"<td>{e(v)}</td>" for v in [cid,q["minimum"],q["maximum"],q["unique_values"],
                          round(q["ac_rms_dbfs"],3),q["informative"],len(c["timestamp_gaps"]),c["pcm_sha256"][:16]])+"</tr>")
    events = "".join(f'<li><strong>{e(a["id"])}</strong>: checked={e(a.get("checked"))}; accepted={e(a.get("accepted_for_fit"))}; {e(a.get("observations"))}; {e(a.get("exclusion_reason") or "")}</li>' for a in doc["events"])
    decisions = "".join(f"<li>{e(t)}</li>" for t in doc["decisions"])
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{e(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    json_link = "time_alignment.json" if output == alignment_path.parent else alignment_path.as_uri()
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShadowTrack — Time Alignment</title><style>body{{font:16px/1.55 system-ui,sans-serif;color:#19252d;background:#f6f7f8;margin:0}}main{{max-width:1400px;margin:auto;padding:28px}}h1{{font-size:30px}}h2{{margin-top:34px}}a{{color:#00657a}}.status{{font-size:24px;color:#93550c}}.scroll{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{border:1px solid #ccd2d5;padding:9px;text-align:left;vertical-align:top}}th{{background:#e4eaed}}img{{width:100%;height:auto}}li{{margin:8px 0}}code{{overflow-wrap:anywhere}}.scroll td{{max-width:370px;overflow-wrap:anywhere}}.definitions{{max-width:1000px}}</style>
<main><h1>ShadowTrack / Time Alignment Engine</h1><p class="status">Acceptance: {e(doc['acceptance_status'])} · связаны {doc['linked_camera_count']} / {count} камер</p>
<p>Глобальное время доступно только внутри поддержанных областей. Неподтверждённые участки возвращают unavailable. Численные интервалы — допустимые границы модели, а не статистические доверительные интервалы.</p>
<p><a href="{e(json_link)}">time_alignment.json</a> · <a href="../../TIME_ALIGNMENT.md">TIME_ALIGNMENT.md</a></p>
<div class="definitions"><h2>Шкалы времени</h2><p><b>raw_media_pts:</b> исходное время MKV. <b>frame_index:</b> индекс декодированного кадра. <b>relative_capture_time_estimate:</b> отдельная локальная оценка с исправлением только проверенных разрывов; физический масштаб не подтверждён. <b>synchronized_global_time:</b> условная общая шкала по проверенным событиям. <b>uncertainty:</b> интервальные границы с источником и допущениями.</p><p>Основа глобальной шкалы: {e(doc['global_time_basis'])}. Время UTC и абсолютное время съёмки не установлены.</p></div>
<h2>Доступность по камерам</h2>{table(['Камера','Header FPS','Frames','Локальных исправлений','Sync status','Кандидаты поддержанных raw-интервалов, s'],camera_rows)}
<p>Точные границы доступности проверяет API с учётом неопределённости пути. Остальная длительность — unsupported. Разрывы PTS не заполняются вымышленными наблюдениями.</p>
{'<img src="global_support.png" alt="Zoomed intervals where global time is available">' if doc['reference_support'] else ''}
<h2>Raw и локально исправленная шкалы</h2><img src="timelines.png" alt="Raw versus locally corrected time, jumps and supported regions">
<h2>Парные модели</h2>{table(['Пара сегментов','Статус','Anchors','Raw model','b: PTS B = a·PTS A + b','Drift ppm','Raw A support, s','RMS residual, s'],pair_rows)}
<p>Поздняя парная модель может быть поддержана в собственном сегменте, но оставаться недоступной в первой глобальной эпохе. Между эпохами неизвестного разрыва точная связь не предполагается.</p>
{'<img src="anchors_residuals.png" alt="Checked anchors, offset envelope and residuals">' if models else '<p>Достаточных проверенных парных anchors нет.</p>'}
<h2>Проверка аудио</h2><p>{e(doc['audio']['status'])}. Декодирование и отдельный audio-only probe не повторяют video probe. PCM каналы проверены по отдельности. Межпоточная A/V задержка не подтверждена; отсутствие общего слышимого события не является отрицательной проверкой синхронности.</p>
{table(['Камера','Min','Max','Unique','AC dBFS','Energy gate','Audio PTS breaks','PCM SHA256 prefix'],audio_rows) if audio else ''}
{'<img src="audio_quality.png" alt="Audio AC energy for five cameras">' if audio else ''}
<h2>Anchors и решения</h2><ul>{events}</ul><ul>{decisions}</ul>
<p>Все исходные MKV, DATA_AUDIT.md и сохранённый video probe остаются неизменными. Detection, Re-ID и global tracking не запускались.</p></main></html>'''
    (output/"report.html").write_text(html,encoding="utf-8")
