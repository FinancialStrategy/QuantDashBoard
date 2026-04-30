const fmtPct = (v,d=2)=> (v===null||v===undefined||Number.isNaN(Number(v))) ? '—' : (Number(v)*100).toFixed(d)+'%';
const fmtNum = (v,d=2)=> (v===null||v===undefined||Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
const fmtMoney = (v)=> (v===null||v===undefined||Number.isNaN(Number(v))) ? '—' : new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(Number(v));
let UNIVERSE = {}; let MAX_TICKERS = 18; let LAST = null;
function setStatus(s){document.getElementById('statusBox').textContent=s}
function classifyKpi(label, value){const l=label.toLowerCase(); if(l.includes('drawdown')||l.includes('var')||l.includes('cvar')) return 'kpi-risk'; if(l.includes('sharpe')||l.includes('return')||l.includes('information')) return 'kpi-good'; if(l.includes('vol')) return 'kpi-warn'; return '';}
function valFormat(label, value){const l=label.toLowerCase(); if(l.includes('value')) return fmtMoney(value); if(l.includes('strategy')) return String(value||'—'); if(l.includes('sharpe')||l.includes('sortino')||l.includes('information')) return fmtNum(value,2); return fmtPct(value,2);}
function kpiCard(label, value, sub=''){return `<div class="kpi-card ${classifyKpi(label,value)}"><div class="kpi-label">${label}</div><div class="kpi-value">${valFormat(label,value)}</div><div class="kpi-sub">${sub||''}</div></div>`}
function formatCell(v, key=''){
  if(v===null||v===undefined||v==='') return '—';
  if(typeof v === 'number'){
    const k=key.toLowerCase();
    if(k.includes('usd')||k.includes('value')||k.includes('nav')) return fmtMoney(v);
    if(k.includes('return')||k.includes('vol')||k.includes('drawdown')||k.includes('var')||k.includes('cvar')||k.includes('weight')||k.includes('rate')||k.includes('contribution')||k.includes('alpha')||k.includes('tracking error')||k.includes('win')) return fmtPct(v,2);
    return fmtNum(v,3);
  }
  return String(v);
}
function renderTable(id, rows){
  const el=document.getElementById(id); if(!rows||!rows.length){el.innerHTML='<div class="callout">No data available.</div>'; return;}
  const cols=Object.keys(rows[0]); let html='<div class="table-wrap"><table class="data-table"><thead><tr>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr></thead><tbody>';
  rows.forEach(r=>{html+='<tr>'+cols.map(c=>`<td>${formatCell(r[c],c)}</td>`).join('')+'</tr>'});
  html+='</tbody></table></div>'; el.innerHTML=html;
}
function plot(id, data, layout){
  const base={paper_bgcolor:'#fff',plot_bgcolor:'#fff',font:{family:'Segoe UI, Arial',color:'#172033',size:12},margin:{l:70,r:40,t:70,b:70},legend:{orientation:'h',y:1.08,x:.5,xanchor:'center'},hovermode:'x unified'};
  Plotly.newPlot(id,data,Object.assign(base,layout||{}),{responsive:true,displayModeBar:false});
}
function sxy(series, ykey='Value'){return {x:(series||[]).map(p=>p.Date), y:(series||[]).map(p=>p[ykey])}}
async function loadUniverse(){
  const r=await fetch('/api/universe'); const js=await r.json(); UNIVERSE=js.universe; MAX_TICKERS=js.max_tickers; buildUniverse();
}
function buildUniverse(){
  const box=document.getElementById('universeBox'); box.innerHTML='';
  Object.entries(UNIVERSE).forEach(([cat,tickers],i)=>{
    const div=document.createElement('div'); div.className='cat';
    div.innerHTML=`<div class="cat-title"><span>${cat}</span><label><input type="checkbox" class="catCheck" data-cat="${cat}"> all</label></div><div class="tickers">${tickers.map(t=>`<label class="tick"><input type="checkbox" class="tickerCheck" data-cat="${cat}" value="${t}"><span>${t}</span></label>`).join('')}</div>`;
    box.appendChild(div);
  });
  document.querySelectorAll('.catCheck').forEach(c=>c.addEventListener('change',e=>{document.querySelectorAll(`.tickerCheck[data-cat="${e.target.dataset.cat}"]`).forEach(x=>x.checked=e.target.checked)}));
  ['US Broad Equity','US Growth / Value','Fixed Income','Real Assets'].forEach(cat=>{[...document.querySelectorAll(`.tickerCheck[data-cat="${cat}"]`)].slice(0,3).forEach(x=>x.checked=true)});
}
function selectedTickers(){return [...document.querySelectorAll('.tickerCheck:checked')].map(x=>x.value).slice(0,MAX_TICKERS)}
function reqPayload(){return {tickers:selectedTickers(),start_date:document.getElementById('startDate').value,initial_capital:Number(document.getElementById('initialCapital').value),expected_return_method:document.getElementById('expectedReturnMethod').value,covariance_method:document.getElementById('covarianceMethod').value,best_strategy_rule:document.getElementById('bestStrategyRule').value,max_weight:Number(document.getElementById('maxWeight').value),max_category_weight:Number(document.getElementById('maxCategory').value),tracking_error_target:Number(document.getElementById('teTarget').value),rolling_window:Number(document.getElementById('rollingWindow').value),mc_simulations:Number(document.getElementById('mcSimulations').value),stress_family:document.getElementById('stressFamily').value,min_severity:Number(document.getElementById('minSeverity').value)}}
async function run(){
  try{
    const tickers=selectedTickers(); if(tickers.length<3){alert('Select at least 3 instruments.'); return;}
    setStatus('Computing institutional analytics from Yahoo Finance...'); document.getElementById('runBtn').disabled=true;
    const res=await fetch('/api/compute',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(reqPayload())});
    const js=await res.json(); if(!res.ok){throw new Error(js.detail?.message || js.detail || JSON.stringify(js));}
    LAST=js; renderAll(js); setStatus(`Done. ${js.meta.observations} daily observations. Compute: ${js.meta.compute_seconds}s`);
  }catch(e){console.error(e); setStatus('Failed.'); alert(String(e));}
  finally{document.getElementById('runBtn').disabled=false;}
}
function renderAll(r){
  document.getElementById('metaLine').textContent=`Benchmark: ${r.meta.benchmark} • RF: ${fmtPct(r.meta.risk_free_rate)} • Yahoo-only • Best: ${r.meta.best_strategy} • ${r.meta.start} → ${r.meta.end}`;
  const k=r.kpis; document.getElementById('kpiGrid').innerHTML=[
    kpiCard('Best Strategy',k['Best Strategy'],r.meta.best_strategy_rule), kpiCard('Annual Return',k['Annual Return']), kpiCard('Volatility',k['Annual Volatility']),
    kpiCard('Sharpe',k['Sharpe']), kpiCard('Max Drawdown',k['Max Drawdown']), kpiCard('Information Ratio',k['Information Ratio']),
    kpiCard('VaR 95 Hist',k['VaR 95 Hist']), kpiCard('CVaR 95 Hist',k['CVaR 95 Hist']), kpiCard('Latest 3M VaR/NAV',k['Latest 3M VaR/NAV']), kpiCard('Final Value',k['Final Value'])
  ].join('');
  document.getElementById('decisionBox').innerHTML=`<b>Selected Strategy:</b> ${r.meta.best_strategy}<br><b>Rule:</b> ${r.meta.best_strategy_rule}<br><b>RF:</b> ${fmtPct(r.meta.risk_free_rate)} fixed. <b>Benchmark:</b> ${r.meta.benchmark} fixed. <b>Data:</b> Yahoo Finance daily only. No fallback or synthetic data used.`;
  renderTable('performanceTable',r.performance_metrics_table); renderTable('strategyTable',r.strategy_table); renderTable('failedTable',r.failed_strategies); renderTable('varTable',r.var_cvar_table); renderTable('riskContributionTable',r.risk_contribution_table); renderTable('stressTable',r.stress_table); renderTable('quantstatsTable',r.quantstats_metrics); renderTable('pcaTable',r.pca_loadings); renderTable('metadataTable',r.metadata_table); renderTable('dataQualityTable',r.data_quality_table); renderTable('pricesTable',r.prices_preview); renderTable('returnsTable',r.returns_preview);
  const eq=sxy(r.series.portfolio_equity), beq=sxy(r.series.benchmark_equity), dd=sxy(r.series.drawdown);
  plot('equityChart',[{x:eq.x,y:eq.y,type:'scatter',mode:'lines',name:'Portfolio',line:{width:3}},{x:beq.x,y:beq.y,type:'scatter',mode:'lines',name:'^GSPC Benchmark',line:{width:2,dash:'dot'}}],{title:'Portfolio vs ^GSPC Equity Curve',yaxis:{title:'USD Value'}});
  plot('drawdownChart',[{x:dd.x,y:dd.y,type:'scatter',mode:'lines',name:'Drawdown',fill:'tozeroy',line:{width:2}}],{title:'Max Drawdown Series',yaxis:{tickformat:'.0%',title:'Drawdown'}});
  plot('allocationChart',[{x:r.weights.map(x=>x.Asset),y:r.weights.map(x=>x.Weight),type:'bar',text:r.weights.map(x=>fmtPct(x.Weight)),textposition:'outside'}],{title:'Selected Strategy Weights',yaxis:{tickformat:'.0%'}});
  plot('strategyScatter',[{x:r.strategy_table.map(x=>x.Volatility),y:r.strategy_table.map(x=>x['Annual Return']),text:r.strategy_table.map(x=>x.Strategy),mode:'markers+text',type:'scatter',textposition:'top center',marker:{size:r.strategy_table.map(x=>Math.max(9,Math.abs(x.Sharpe||0)*8))}}],{title:'Strategy Risk / Return Map',xaxis:{title:'Volatility',tickformat:'.0%'},yaxis:{title:'Annual Return',tickformat:'.0%'}});
  plot('riskContributionChart',[{x:r.risk_contribution_table.map(x=>x.Asset),y:r.risk_contribution_table.map(x=>x['Contribution %']),type:'bar',text:r.risk_contribution_table.map(x=>fmtPct(x['Contribution %'])),textposition:'outside'}],{title:'Risk Contributions',yaxis:{tickformat:'.0%'}});
  const rvn={x:r.series.rolling_var_nav.map(x=>x.Date),y:r.series.rolling_var_nav.map(x=>x['Rolling 3M VaR/NAV'])};
  plot('varNavChart',[{x:rvn.x,y:rvn.y,type:'scatter',mode:'lines',name:'3M VaR/NAV',line:{width:2}}],{title:'Rolling 3-Month VaR / NAV Ratio',yaxis:{tickformat:'.0%'}});
  const rb=sxy(r.series.rolling_beta), rs=sxy(r.series.rolling_sharpe), rvol=sxy(r.series.rolling_volatility);
  plot('rollingBetaChart',[{x:rb.x,y:rb.y,type:'scatter',mode:'lines',name:'Rolling Beta'}],{title:'Rolling Beta vs ^GSPC',yaxis:{title:'Beta'}});
  plot('rollingSharpeChart',[{x:rs.x,y:rs.y,type:'scatter',mode:'lines',name:'Rolling Sharpe'}],{title:'Rolling Sharpe',yaxis:{title:'Sharpe'}});
  plot('rollingVolChart',[{x:rvol.x,y:rvol.y,type:'scatter',mode:'lines',name:'Rolling Volatility'}],{title:'Rolling Annualized Volatility',yaxis:{tickformat:'.0%'}});
  const dr=r.series.daily_returns||[]; plot('returnsHistogram',[{x:dr.map(x=>x.Portfolio),type:'histogram',nbinsx:70,name:'Portfolio Daily Returns'}],{title:'Portfolio Daily Returns Distribution',xaxis:{tickformat:'.1%'}});
  document.getElementById('stressKpis').innerHTML=Object.entries(r.stress_kpis).map(([kk,v])=>kpiCard(kk,v)).join('');
  plot('stressChart',[{x:r.stress_table.map(x=>x['Relative Return']),y:r.stress_table.map(x=>x.Scenario),type:'bar',orientation:'h',name:'Relative Return'}],{title:'Stress Scenario Ranking',xaxis:{tickformat:'.0%'}});
  plot('pcaVarianceChart',[{x:r.pca_variance.map(x=>x.Component),y:r.pca_variance.map(x=>x['Explained Variance']),type:'bar'}],{title:'PCA Explained Variance',yaxis:{tickformat:'.0%'}});
  plot('pcaLoadingsChart',[{x:r.pca_loadings.map(x=>x.Asset),y:r.pca_loadings.map(x=>x.PC1),type:'bar'}],{title:'PC1 Loadings'});
}
document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active-panel'));document.getElementById('tab-'+btn.dataset.tab).classList.add('active-panel');setTimeout(()=>window.dispatchEvent(new Event('resize')),100)}));
document.getElementById('runBtn').addEventListener('click',run); loadUniverse();
