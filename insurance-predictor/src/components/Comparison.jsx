import { useState, useEffect } from 'react'

function MetricRow({ label, value }) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-gray-50 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-sm font-medium text-gray-900">{value}</span>
    </div>
  )
}

export default function Comparison() {
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    fetch('/metrics.json').then(r => r.json()).then(setMetrics)
  }, [])

  return (
    <div className="space-y-4">

      {/* Analysis */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
        <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">Analysis</p>
        <ul className="space-y-1.5">
          {[
            'The neural network achieved a lower mean absolute error than the random forest, though R² and RMSE were nearly identical between both models.',
            'Smoking status, age, and BMI were the strongest predictors of insurance cost, showing the largest impact on model error when permuted.',
          ].map((point, i) => (
            <li key={i} className="flex gap-2 text-sm text-gray-600">
              <span className="text-gray-300 mt-0.5">-</span>
              <span>{point}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-gray-900">Neural Network</h2>
            <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">Keras / TensorFlow</span>
          </div>
          <div className="bg-gray-50 rounded-xl px-4 py-2">
            {metrics?.nn ? (
              <>
                <MetricRow label="Mean Absolute Error" value={`$${metrics.nn.mae.toLocaleString()}`} />
                <MetricRow label="Root Mean Squared Error" value={`$${metrics.nn.rmse.toLocaleString()}`} />
                <MetricRow label="R² Score" value={metrics.nn.r2.toFixed(3)} />
              </>
            ) : <div className="py-2 text-xs text-gray-400">Loading...</div>}
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5 flex flex-col gap-4">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-gray-900">Random Forest</h2>
            <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">scikit-learn</span>
          </div>
          <div className="bg-gray-50 rounded-xl px-4 py-2">
            {metrics?.rf ? (
              <>
                <MetricRow label="Mean Absolute Error" value={`$${metrics.rf.mae.toLocaleString()}`} />
                <MetricRow label="Root Mean Squared Error" value={`$${metrics.rf.rmse.toLocaleString()}`} />
                <MetricRow label="R² Score" value={metrics.rf.r2.toFixed(3)} />
              </>
            ) : <div className="py-2 text-xs text-gray-400">Loading...</div>}
          </div>
        </div>
      </div>

      {/* Feature importance — side by side */}
      <div className="grid grid-cols-2 gap-4">
        {[
          { label: 'Feature Importance (Neural Network)', src: '/nn_feature_importance.png' },
          { label: 'Feature Importance (Random Forest)', src: '/rf_feature_importance.png' },
        ].map(({ label, src }) => (
          <div key={src} className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
            <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-3">{label}</p>
            <img src={src} alt={label} className="w-full rounded-xl border border-gray-100" />
          </div>
        ))}
      </div>

    </div>
  )
}
