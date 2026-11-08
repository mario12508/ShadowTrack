import {test,expect} from '@playwright/test';

test('actual FULL multi-view episode has consistent IDs in API, Gantt and Monitor',async({page,request})=>{
 const data=await (await request.get('/api/gantt')).json();
 const row=data.rows.find((r:any)=>r.status==='PROVISIONAL'&&['3','4','5'].every(c=>r.segments.some((s:any)=>s.camera_id===c)));
 expect(row).toBeTruthy();
 await page.goto('/timeline');
 await page.getByRole('textbox',{name:'Поиск эпизода'}).fill(row.display_label);
 await expect(page.locator('.gantt-row')).toHaveCount(1);
 await expect(page.locator('.gantt-label')).toContainText(row.display_label);
 const button=page.locator('.gantt-segment').filter({hasText:''}).first();await button.click();
 await expect(page.locator('.track-selection')).toContainText(row.display_label,{timeout:20000});
 await expect(page.locator('.identity-badge.PROVISIONAL').first()).toBeVisible();
 await expect(page.locator('.clock-explainer')).toContainText('исходная шкала записей');
 const member=row.segments.find((s:any)=>s.camera_id==='3');
 await page.getByRole('button',{name:'Камера 3',exact:true}).click();
 await expect(page.locator('.track-selection')).toContainText(row.display_label,{timeout:20000});
 const detail=await (await request.get('/api/tracks/'+member.track_id)).json();
 expect(detail.global_label).toBe(row.display_label);expect(detail.global_status).toBe('PROVISIONAL');
 await page.getByRole('link',{name:'Gantt',exact:true}).first().click();
 await expect(page.locator('.gantt-row.selected .gantt-label')).toContainText(row.display_label);
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/mtmc-gantt.png',fullPage:true});
});

test('all episode rows are available with virtualization and explicit estimated time',async({page,request})=>{
 const data=await (await request.get('/api/gantt')).json();
 expect(data.rows.length).toBeGreaterThan(200);
 await page.goto('/timeline');await expect(page.getByTestId('gantt')).toBeVisible();
 await expect.poll(()=>page.locator('.gantt-row').count()).toBeLessThanOrEqual(10);
 await page.getByTestId('gantt').evaluate((e:HTMLElement)=>{e.scrollTop=e.scrollHeight});
 await expect.poll(()=>page.locator('.gantt-row').count()).toBeGreaterThan(0);
 await page.getByRole('button',{name:'Увеличить масштаб'}).click();
 await expect(page.getByText('1.5×',{exact:true})).toBeVisible();
 await page.getByRole('combobox',{name:'Статус эпизода'}).selectOption('CONFIRMED');
 await expect(page.locator('.gantt-row')).toHaveCount(0);
 await expect(page.getByText(/Estimated timeline/)).toBeVisible();
});

test('real minimap uses moving approximate dots with consistent IDs',async({page,request})=>{
 const floor=await (await request.get('/api/floor?time=1030')).json();expect(floor.markers.length).toBeGreaterThan(0);expect(floor.markers.every((m:any)=>m.position?.length===2&&m.floor_zone?.radius_normalized>0&&m.localization_status==='SCHEMATIC_PROJECTED')).toBeTruthy();
 await page.goto('/monitor?camera=4&time=1030&clock=estimated&gtime=1030');
 await page.getByRole('button',{name:'Люди',exact:true}).click();
 await expect(page.locator('.floor-observations>button').first()).toBeVisible();
 await expect(page.locator('.person-pin').first()).toBeVisible();await expect(page.locator('.floor-zone')).toHaveCount(0);
 await expect(page.locator('.floor-accuracy')).toContainText('положение на схеме приблизительное');
 await page.locator('.floor-observations>button').first().click();await expect(page.locator('.track-selection')).toBeVisible();
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/mtmc-floor.png',fullPage:true});
});

test('candidate review is explicit and new workspace screens work on mobile',async({page})=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('/associations');await expect(page.locator('.candidate-card').first()).toBeVisible();
 await expect(page.getByRole('button',{name:'Подтвердить',exact:true}).first()).toBeDisabled();
 await expect(page.locator('.association-summary')).toContainText('0 проверенных вручную');
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/mtmc-review.png',fullPage:true});
 await page.goto('/geometry');await expect(page.getByText(/Сначала точка на полу/)).toBeVisible();
 await page.setViewportSize({width:393,height:852});
 for(const route of ['/timeline','/associations','/geometry']){
  await page.goto(route);await expect(page.locator('h2')).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBeTruthy();
 }
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/mtmc-mobile.png',fullPage:true});
 expect(errors).toEqual([]);
});


test('reported corridor handoff uses the same global track on camera 4 and 5',async({page,request})=>{
 const a=await (await request.get('/api/overlays/4?start=19&end=21')).json(),b=await (await request.get('/api/overlays/5?start=23&end=25')).json();
 const aa=a.frames.flatMap((f:any)=>f.tracks).filter((t:any)=>t.display_id==='C4-T001'),bb=b.frames.flatMap((f:any)=>f.tracks).filter((t:any)=>t.display_id==='C5-T006');
 expect(aa.length).toBeGreaterThan(10);expect(bb.length).toBeGreaterThan(10);
 const ids=new Set([...aa,...bb].map((t:any)=>t.global_episode_id));expect(ids.size).toBe(1);const id=aa[0].global_episode_id,label=aa[0].global_label;
 const catalog=await (await request.get('/api/global-tracks?q='+encodeURIComponent(label))).json();expect(catalog.total).toBe(1);expect(catalog.items[0].camera_ids).toEqual(expect.arrayContaining(['4','5']));
 for(const [camera,time] of [['4',20],['5',23]] as const){
  const floor=await (await request.get('/api/floor?time='+time)).json();expect(floor.markers.filter((m:any)=>m.global_episode_id===id)).toHaveLength(1);
  await page.goto(`/monitor?camera=${camera}&time=${time}&global=${id}`);await page.getByRole('button',{name:'Люди',exact:true}).click();
  const video=page.locator('.camera-panel video');await expect.poll(()=>video.evaluate((v:HTMLVideoElement,expected:number)=>v.readyState>=2&&Math.abs(v.currentTime-expected)<.2,time),{timeout:20000}).toBeTruthy();
  await expect.poll(()=>page.locator('.camera-panel canvas').evaluate((c:HTMLCanvasElement)=>{const p=c.getContext('2d')!.getImageData(0,0,c.width,c.height).data;let n=0;for(let i=3;i<p.length;i+=4)if(p[i])n++;return n>150}),{timeout:15000}).toBeTruthy();
  await expect(page.getByRole('button',{name:label+' на карте'})).toBeVisible();
  await page.screenshot({path:`../private_outputs/mvp_browser_smoke/corridor-handoff-${camera}.png`,fullPage:true});
 }
});

test('archive groups local fragments into one global card and numbers start at one',async({page,request})=>{
 const raw=await (await request.get('/api/tracks?q=C3-T011')).json();const label=raw.items[0].global_label;
 const catalog=await (await request.get('/api/global-tracks?q='+encodeURIComponent(label))).json();expect(catalog.total).toBe(1);expect(catalog.items[0].fragment_count).toBeGreaterThanOrEqual(2);
 const first=await (await request.get('/api/global-tracks?limit=24')).json();expect(first.items.map((r:any)=>r.global_label)).toEqual(Array.from({length:24},(_,i)=>'ID '+String(i+1).padStart(3,'0')));
 await page.goto('/tracks');await page.getByPlaceholder('Найти глобальный ID…').fill(label);
 await expect(page.locator('.track-card')).toHaveCount(1);await expect(page.locator('.track-card h3')).toHaveText(label);
 await expect(page.locator('.track-card-footer')).toContainText('фрагм.');await expect.poll(()=>page.locator('.track-card img').evaluate((img:HTMLImageElement)=>img.complete&&img.naturalWidth>0),{timeout:15000}).toBeTruthy();await page.screenshot({path:'../private_outputs/mvp_browser_smoke/global-archive.png',fullPage:true});
 // Real reported torso/full-body and front/back fragments form one archive
 // entity. These are regression assertions, never inputs to association.
 const ids=new Set<string>();
 for(const source of ['C3-T001','C3-T002','C5-T001','C5-T002','C5-T003']){
  const data=await (await request.get('/api/tracks?q='+source)).json();
  const track=data.items.find((t:any)=>t.display_id===source);expect(track).toBeTruthy();ids.add(track.global_episode_id);
 }
 expect(ids.size).toBe(1);
 await page.getByPlaceholder('Найти глобальный ID…').fill('ID 002');
 await expect(page.locator('.track-card')).toHaveCount(1);await expect(page.locator('.track-card h3')).toHaveText('ID 002');
 await expect(page.locator('.track-card-footer')).toContainText('фрагм.');
 const woman=await (await request.get('/api/global-tracks?q=ID%20002')).json();expect(woman.items[0].fragment_count).toBeGreaterThanOrEqual(5);
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/woman-single-archive-card.png',fullPage:true});
});
