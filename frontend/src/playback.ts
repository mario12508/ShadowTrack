/** Native source time only. Calibration and API availability never drive seeking. */
export function sourceTime(requested:number,duration:number){
 return Math.max(0,Math.min(Math.max(0,duration-.02),Number.isFinite(requested)?requested:0));
}
