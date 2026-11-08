import {test,expect} from '@playwright/test';

test('map and people modes differ, preserve native video and clear global selection',async({page})=>{
 const requests:string[]=[];page.on('request',r=>requests.push(r.url()));
 await page.goto('/monitor?camera=3&time=129');
 await expect(page.getByTestId('camera-map')).toBeVisible();
 await expect(page.locator('.map-pin')).toHaveCount(5);
 await expect(page.locator('.person-pin,.floor-observations')).toHaveCount(0);
 const video=page.locator('.camera-panel video');
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.readyState>=2&&Math.abs(v.currentTime-129)<.2)).toBeTruthy();
 await video.evaluate((v:HTMLVideoElement)=>{(window as any).__samePlayer=v;(v as any).__seeks=0;v.addEventListener('seeking',()=>{(v as any).__seeks++})});
 await page.getByRole('button',{name:'Воспроизвести',exact:true}).click();
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(130);
 expect(requests.some(url=>url.includes('/api/floor?'))).toBeFalsy();
 await page.getByRole('button',{name:'Люди',exact:true}).click();
 await expect(page.getByTestId('people-map')).toBeVisible();await expect(page.locator('.map-pin')).toHaveCount(0);
 await expect(page.locator('.person-pin').first()).toBeVisible();
 expect(await video.evaluate(v=>v===(window as any).__samePlayer)).toBeTruthy();
 await expect.poll(()=>video.evaluate((v:HTMLVideoElement)=>v.currentTime)).toBeGreaterThan(135);
 expect(await video.evaluate(v=>(v as any).__seeks)).toBe(0);
 expect(requests.filter(url=>url.includes('/poster?time='))).toHaveLength(0);
 await page.getByRole('button',{name:'Пауза',exact:true}).click();
 await page.locator('.person-pin').first().click();await expect(page.locator('.track-selection')).toBeVisible();
 await page.getByRole('button',{name:'Снять выбор',exact:true}).click();
 await expect(page.locator('.track-selection,.person-pin.selected')).toHaveCount(0);
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/people-mode.png',fullPage:true});
 await page.getByRole('button',{name:'Карта',exact:true}).click();
 await expect(page.locator('.map-pin')).toHaveCount(5);await expect(page.locator('.person-pin')).toHaveCount(0);
 await page.setViewportSize({width:393,height:852});
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1)).toBeTruthy();
 await page.screenshot({path:'../private_outputs/mvp_browser_smoke/camera-mode-mobile.png',fullPage:true});
});

test('episode count, cross-camera subset and export agree without future leakage',async({page,request})=>{
 const total=await (await request.get('/api/episode-counts')).json();
 const early=await (await request.get('/api/episode-counts?time=0')).json();
 const stats=await (await request.get('/api/stats')).json();
 expect(total.total_episodes).toBe(stats.global_episodes);
 expect(total.multi_camera_episodes+total.single_camera_episodes).toBe(total.total_episodes);
 expect(total.active+total.lost+total.terminated).toBe(total.started_episodes);
 expect(early.started_episodes).toBeLessThan(total.started_episodes);
 const csv=await (await request.get('/api/exports/episodes.csv')).text();expect(csv.trim().split('\n')).toHaveLength(total.total_episodes+1);
 await page.goto('/monitor?time=0');await expect(page.locator('.episode-counter strong').first()).toHaveText(String(total.total_episodes));
 await expect(page.locator('.episode-counter strong').nth(1)).toHaveText(String(early.started_episodes));
 await page.goto('/analytics');await expect(page.locator('.episode-totals')).toContainText(String(total.multi_camera_episodes));
 await expect(page.getByRole('link',{name:'Перечень эпизодов · CSV'})).toBeVisible();
});

test('failed media displays a recoverable error instead of an unexplained black player',async({page})=>{
 await page.route('**/api/media/**',route=>route.abort());
 await page.goto('/monitor?camera=3&time=129');
 await expect(page.locator('.video-error')).toBeVisible({timeout:20000});
 await page.unroute('**/api/media/**');
 await page.getByRole('button',{name:'Повторить загрузку',exact:true}).click();
 await expect.poll(()=>page.locator('.camera-panel video').evaluate((v:HTMLVideoElement)=>v.readyState),{timeout:20000}).toBeGreaterThanOrEqual(2);
 await expect(page.locator('.video-error')).toHaveCount(0);
});
