import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Dataset } from '../types';
import { Info, AlertCircle, Table as TableIcon, BarChart3, Hash } from 'lucide-react';

interface DataExplorationProps {
  dataset: Dataset;
}

export const DataExploration: React.FC<DataExplorationProps> = ({ dataset }) => {
  const { data, features, target } = dataset;

  // Summary Stats
  const summary = features.map(f => {
    const values = data.map(d => parseFloat(d[f])).filter(v => !isNaN(v));
    const min = values.length > 0 ? Math.min(...values) : 0;
    const max = values.length > 0 ? Math.max(...values) : 0;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const missing = data.length - values.length;
    return { feature: f, min, max, mean, missing };
  });

  // Correlation Matrix (Simplified for top 5 features)
  const topFeatures = features.slice(0, 5);
  const correlationMatrix = topFeatures.map(f1 => {
    return topFeatures.map(f2 => {
      const v1 = data.map(d => parseFloat(d[f1]));
      const v2 = data.map(d => parseFloat(d[f2]));
      const n = v1.length;
      const sum1 = v1.reduce((a, b) => a + b, 0);
      const sum2 = v2.reduce((a, b) => a + b, 0);
      const sum1Sq = v1.reduce((a, b) => a + b * b, 0);
      const sum2Sq = v2.reduce((a, b) => a + b * b, 0);
      const pSum = v1.reduce((a, b, i) => a + b * v2[i], 0);
      const num = pSum - (sum1 * sum2 / n);
      const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
      return den === 0 ? 0 : num / den;
    });
  });

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard icon={<TableIcon className="text-blue-500" />} label="Total Samples" value={data.length} />
        <StatCard icon={<BarChart3 className="text-emerald-500" />} label="Features" value={features.length} />
        <StatCard icon={<Hash className="text-amber-500" />} label="Target" value={target || 'N/A'} />
        <StatCard icon={<AlertCircle className="text-red-500" />} label="Missing Values" value={summary.reduce((a, b) => a + b.missing, 0)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Feature Distributions */}
        <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm">
          <h4 className="text-sm font-bold uppercase text-black/40 mb-6 flex items-center gap-2">
            <BarChart3 className="w-4 h-4" /> Feature Distributions
          </h4>
          <div className="space-y-6">
            {features.slice(0, 3).map(f => {
              const values = data.map(d => parseFloat(d[f])).filter(v => !isNaN(v));
              const bins = 10;
              const min = values.length > 0 ? Math.min(...values) : 0;
              const max = values.length > 0 ? Math.max(...values) : 0;
              const step = (max - min) / bins;
              const histogram = Array.from({ length: bins }, (_, i) => {
                const start = min + i * step;
                const end = start + step;
                const count = values.filter(v => v >= start && v < (i === bins - 1 ? end + 0.1 : end)).length;
                return { bin: `${start.toFixed(1)}-${end.toFixed(1)}`, count };
              });

              return (
                <div key={f} className="h-40">
                  <p className="text-xs font-bold mb-2">{f}</p>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={histogram}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                      <XAxis dataKey="bin" fontSize={8} tickLine={false} axisLine={false} />
                      <YAxis fontSize={8} tickLine={false} axisLine={false} />
                      <Tooltip 
                        contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                        labelStyle={{ fontWeight: 'bold', fontSize: '10px' }}
                      />
                      <Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              );
            })}
          </div>
        </div>

        {/* Correlation Heatmap */}
        <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm">
          <h4 className="text-sm font-bold uppercase text-black/40 mb-6 flex items-center gap-2">
            <Info className="w-4 h-4" /> Feature Correlation
          </h4>
          <div className="relative aspect-square">
            <div className="absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${topFeatures.length}, 1fr)` }}>
              {correlationMatrix.flat().map((val, i) => {
                const row = Math.floor(i / topFeatures.length);
                const col = i % topFeatures.length;
                const opacity = Math.abs(val);
                const color = val > 0 ? `rgba(16, 185, 129, ${opacity})` : `rgba(239, 68, 68, ${opacity})`;
                return (
                  <div 
                    key={i} 
                    className="flex items-center justify-center text-[8px] font-bold border border-black/5 transition-transform hover:scale-105 hover:z-10 cursor-help"
                    style={{ backgroundColor: color, color: opacity > 0.5 ? 'white' : 'black' }}
                    title={`${topFeatures[row]} vs ${topFeatures[col]}: ${val.toFixed(2)}`}
                  >
                    {val.toFixed(2)}
                  </div>
                );
              })}
            </div>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {topFeatures.map((f, i) => (
              <span key={i} className="text-[8px] font-bold px-2 py-1 bg-black/5 rounded uppercase">{i}: {f}</span>
            ))}
          </div>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white p-6 rounded-3xl border border-black/5 shadow-sm overflow-x-auto">
        <h4 className="text-sm font-bold uppercase text-black/40 mb-6">Statistical Summary</h4>
        <table className="w-full text-left text-xs">
          <thead>
            <tr className="border-b border-black/5">
              <th className="pb-4 font-bold text-black/40 uppercase">Feature</th>
              <th className="pb-4 font-bold text-black/40 uppercase">Mean</th>
              <th className="pb-4 font-bold text-black/40 uppercase">Min</th>
              <th className="pb-4 font-bold text-black/40 uppercase">Max</th>
              <th className="pb-4 font-bold text-black/40 uppercase">Missing</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-black/5">
            {summary.map(s => (
              <tr key={s.feature} className="hover:bg-black/[0.02] transition-colors">
                <td className="py-4 font-bold">{s.feature}</td>
                <td className="py-4 font-mono">{s.mean.toFixed(2)}</td>
                <td className="py-4 font-mono">{s.min.toFixed(2)}</td>
                <td className="py-4 font-mono">{s.max.toFixed(2)}</td>
                <td className="py-4">
                  <span className={`px-2 py-1 rounded-full font-bold text-[10px] ${s.missing > 0 ? 'bg-red-100 text-red-600' : 'bg-emerald-100 text-emerald-600'}`}>
                    {s.missing}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

function StatCard({ icon, label, value }: { icon: React.ReactNode, label: string, value: string | number }) {
  return (
    <div className="bg-white p-6 rounded-2xl border border-black/5 shadow-sm">
      <div className="flex items-center gap-3 mb-2">
        {icon}
        <span className="text-[10px] font-bold uppercase text-black/40">{label}</span>
      </div>
      <p className="text-2xl font-bold tracking-tight">{value}</p>
    </div>
  );
}
