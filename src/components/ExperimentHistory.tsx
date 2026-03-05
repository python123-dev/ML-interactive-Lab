import React from 'react';
import { HistoryItem } from '../types';
import { 
  Trash2, RefreshCw, ChevronRight, Calendar, Database, Cpu, 
  CheckCircle2, AlertCircle, BarChart2 
} from 'lucide-react';

interface ExperimentHistoryProps {
  history: HistoryItem[];
  onDelete: (id: number) => void;
  onReRun: (item: HistoryItem) => void;
  onCompare: (items: HistoryItem[]) => void;
}

export const ExperimentHistory: React.FC<ExperimentHistoryProps> = ({ history, onDelete, onReRun, onCompare }) => {
  const [selectedIds, setSelectedIds] = React.useState<number[]>([]);

  const toggleSelection = (id: number) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold">Experiment History</h3>
        {selectedIds.length > 1 && (
          <button 
            onClick={() => onCompare(history.filter(h => selectedIds.includes(h.id)))}
            className="px-4 py-2 bg-emerald-600 text-white rounded-full text-xs font-bold uppercase tracking-widest hover:bg-emerald-700 transition-all flex items-center gap-2"
          >
            <BarChart2 className="w-4 h-4" /> Compare Selected ({selectedIds.length})
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4">
        {history.length === 0 ? (
          <div className="p-12 text-center bg-white rounded-3xl border border-black/5 border-dashed">
            <Database className="w-12 h-12 text-black/10 mx-auto mb-4" />
            <p className="text-black/40 font-medium">No experiments yet. Train a model to see it here!</p>
          </div>
        ) : (
          history.map(item => (
            <div 
              key={item.id} 
              className={`group relative bg-white p-6 rounded-3xl border transition-all hover:shadow-xl ${selectedIds.includes(item.id) ? 'border-emerald-600 ring-1 ring-emerald-600' : 'border-black/5'}`}
            >
              <div className="flex items-start gap-4">
                <div 
                  onClick={() => toggleSelection(item.id)}
                  className={`mt-1 w-5 h-5 rounded-full border-2 flex items-center justify-center cursor-pointer transition-colors ${selectedIds.includes(item.id) ? 'bg-emerald-600 border-emerald-600 text-white' : 'border-black/10 hover:border-emerald-600'}`}
                >
                  {selectedIds.includes(item.id) && <CheckCircle2 className="w-3 h-3" />}
                </div>

                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-bold px-2 py-0.5 bg-emerald-100 text-emerald-700 rounded uppercase tracking-tighter">
                        {item.algo_name}
                      </span>
                      <span className="text-[10px] font-bold px-2 py-0.5 bg-blue-100 text-blue-700 rounded uppercase tracking-tighter">
                        {item.dataset_name}
                      </span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-[10px] font-bold text-black/30 flex items-center gap-1">
                        <Calendar className="w-3 h-3" /> {new Date(item.created_at).toLocaleDateString()}
                      </span>
                      <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button 
                          onClick={() => onReRun(item)}
                          className="p-2 bg-emerald-50 text-emerald-600 rounded-lg hover:bg-emerald-100 transition-colors"
                          title="Re-run experiment"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </button>
                        <button 
                          onClick={() => onDelete(item.id)}
                          className="p-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
                          title="Delete experiment"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    {Object.entries(item.metrics).filter(([k, v]) => typeof v === 'number' && k !== 'inertia').map(([key, val]: [string, any]) => (
                      <div key={key} className="p-3 bg-stone-50 rounded-xl border border-black/5">
                        <p className="text-[8px] font-bold uppercase text-black/40 mb-1">{key}</p>
                        <p className="text-sm font-bold">{val.toFixed(3)}</p>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    {item.features.map(f => (
                      <span key={f} className="text-[8px] font-bold px-2 py-0.5 bg-black/5 text-black/40 rounded uppercase">
                        {f}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
