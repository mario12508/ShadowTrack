"""Offline report; all values and filenames are HTML-escaped."""

from html import escape
from pathlib import Path
import shutil


def write_report(report: dict, output: Path) -> None:
    def e(value):
        return escape(str(value))

    rows, sections = [], []
    for c in report["cameras"]:
        t = c.get("timestamps", {})
        values = [Path(c["file"]).name, c["status"], c.get("codec"),
                  f"{c.get('width')} × {c.get('height')}", c.get("r_frame_rate"),
                  f"{t.get('observed_fps', 0):.6f}" if t.get("observed_fps") else "—",
                  t.get("decoded_frames"), c.get("container_duration_seconds"),
                  t.get("first_pts_seconds"), t.get("last_pts_seconds")]
        rows.append("<tr>" + "".join(f"<td>{e(v)}</td>" for v in values) + "</tr>")
        body = f"<p>{e(c.get('error', ''))}</p>"
        if c.get("contact_sheet"):
            body += f'<a href="{e(c["contact_sheet"])}"><img class="sheet" src="{e(c["contact_sheet"])}" alt="Contact sheet {e(Path(c["file"]).name)}"></a>'
            body += '<details><summary>Полноразмерные preview и фактические PTS</summary><div class="gallery">'
            for sample in c["previews"]:
                link = c["key"] + "/" + sample["path"]
                body += f'<a href="{e(link)}"><img loading="lazy" src="{e(link)}" alt="PTS {sample["actual_pts_seconds"]:.3f} s"><span>PTS {sample["actual_pts_seconds"]:.3f} s</span></a>'
            body += "</div></details>"
        body += f'<p><a href="{c["key"]}/timestamps/probe.json">ffprobe</a> · <a href="{c["key"]}/timestamps/frames.json">Все timestamps</a></p>'
        sections.append(f'<section id="{c["key"]}"><h2>{e(Path(c["file"]).name)}</h2>{body}</section>')
    issues = "".join(f'<li><strong>{e(i["code"])}</strong>'
                     f'{" — " + e(Path(i["file"]).name) if i.get("file") else ""}: '
                     f'{e(i["message"])}</li>' for i in report["issues"])
    plans = ""
    for i, item in enumerate(report["metadata"]["files"]):
        if item.get("kind") == "raster_plan_candidate":
            source = Path(item["file"])
            name = f"metadata_{i}{source.suffix.lower()}"
            shutil.copyfile(source, output / name)
            plans += f'<a href="{name}"><img class="plan" src="{name}" alt="{e(source.name)}"></a>'
    headings = ["Файл", "Статус", "Codec", "Размер", "FPS header", "FPS PTS", "Frames", "Duration, s", "First PTS, s", "Last PTS, s"]
    review = ""
    if report.get("analyst_review"):
        review = '<h2>Визуальное исследование</h2><p>Авторские наблюдения и гипотезы; привязаны к SHA256 исходных файлов. Не являются калибровкой или разметкой ID.</p>'
        for section in report["analyst_review"]["sections"]:
            review += f'<h3>{e(section["title"])}</h3>' + ''.join(f'<p>{e(p)}</p>' for p in section["paragraphs"])
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ShadowTrack — аудит данных</title><style>
body{{font:16px/1.55 system-ui,sans-serif;background:#f5f6f7;color:#18232c;margin:0}}main{{max-width:1380px;margin:auto;padding:32px}}
h1{{font-size:32px}}h2{{margin-top:40px}}a{{color:#00647a}}.notice{{border-left:4px solid #b16b14;padding:12px 20px;background:#fff4df}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}th,td{{text-align:left;border:1px solid #ccd3d8;padding:9px;white-space:nowrap}}
th{{background:#e5ebed}}.sheet{{width:100%;height:auto}}.plan{{max-height:700px;max-width:100%;object-fit:contain}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:15px}}.gallery img{{width:100%}}details{{margin:16px 0}}li{{margin:8px 0}}footer{{margin-top:40px;font-size:13px;color:#52616c}}
</style><main><h1>ShadowTrack / Аудит исходных данных</h1>
<p>Статус: <strong>{e(report['status'])}</strong> · Создан: {e(report['created_at_utc'])} · <a href="stats.json">JSON stats</a></p>
<p class="notice">PTS — время медиапотока, а не часы в кадре. Межкамерная синхронизация не подтверждена; поправки времени не применялись. ID людей не размечались.</p>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{x}</th>' for x in headings)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>Проверки и ограничения</h2><ul>{issues or '<li>Автоматические проверки не выявили ошибок.</li>'}</ul>
<p>Frames: число кадров полного декодирования ffprobe. FPS PTS: (N − 1)/(last PTS − first PTS). Duration — контейнер, может включать аудио. Preview распределены по диапазону видеопотока; подписи показывают фактический PTS и время от первого кадра.</p>
<p>Ручные наблюдения сцен, гипотезы перекрытий и исследование часов: <a href="../../DATA_AUDIT.md">DATA_AUDIT.md</a> (при стандартном расположении отчёта).</p>
{review}<h2>Реальные metadata</h2>{plans or '<p>Читаемый план не найден.</p>'}
{''.join(sections)}<footer>Локальный отчёт. Нет внешних ресурсов, загрузки видео в сеть или ручной разметки ID. Повторный запуск обновляет индекс; источники остаются неизменными.</footer></main></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")
