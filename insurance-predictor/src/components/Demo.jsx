import { useState, useEffect, useRef } from 'react'

const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true'

async function loadTfjs() {
  const tf = await import('@tensorflow/tfjs')
  const [model, scaler, means] = await Promise.all([
    tf.loadLayersModel('/model/model.json'),
    fetch('/scaler_params.json').then(r => r.json()),
    fetch('/feature_means.json').then(r => r.json()),
  ])
  return { tf, model, scaler, means }
}

export default function Demo() {
  const tfjsRef = useRef(null)
  const [modelStatus, setModelStatus] = useState(DEMO_MODE ? 'loading' : 'ready')
  const [form, setForm] = useState({ age: '30', bmi: '25.0', children: 0, smoker: 'no' })
  const [result, setResult] = useState(null)
  const [predicting, setPredicting] = useState(false)

  useEffect(() => {
    if (!DEMO_MODE) return
    loadTfjs()
      .then(refs => { tfjsRef.current = refs; setModelStatus('ready') })
      .catch(e => { console.error(e); setModelStatus('error') })
  }, [])

  function set(key, value) {
    setForm(f => ({ ...f, [key]: value }))
    setResult(null)
  }

  async function predict() {
    setPredicting(true)
    try {
      let value
      if (DEMO_MODE) {
        const { tf, model, scaler, means } = tfjsRef.current
        const input = { ...means, age: parseFloat(form.age), bmi: parseFloat(form.bmi), children: form.children, smoker: form.smoker === 'yes' ? 1 : 0 }
        const scaled = scaler.feature_names.map((f, i) => (input[f] - scaler.mean[i]) / scaler.scale[i])
        const tensor = tf.tensor2d([scaled])
        const pred = model.predict(tensor)
        value = (await pred.data())[0]
        tensor.dispose()
        pred.dispose()
      } else {
        const res = await fetch('http://localhost:5000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...form, age: parseFloat(form.age), bmi: parseFloat(form.bmi) }),
        })
        const data = await res.json()
        value = data.charges
      }
      setResult(Math.max(0, value))
    } catch (e) {
      console.error(e)
    } finally {
      setPredicting(false)
    }
  }

  const inputClass = 'w-full border border-gray-200 rounded-xl px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-300 transition'
  const labelClass = 'block text-xs font-medium text-gray-500 mb-1'

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <h1 className="text-base font-semibold text-gray-900 mb-1">Insurance Cost Predictor</h1>
        <p className="text-sm text-gray-500 mb-5">Enter your details to get an estimated annual health insurance charge.</p>

        {modelStatus === 'loading' && (
          <div className="flex items-center gap-2 text-sm text-gray-400 mb-5">
            <svg className="animate-spin h-4 w-4 text-gray-400" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
            </svg>
            Loading model...
          </div>
        )}
        {modelStatus === 'error' && (
          <div className="text-sm text-red-500 mb-5">Failed to load model.</div>
        )}

        <div className="grid grid-cols-2 gap-4 mb-5">
          <div>
            <label className={labelClass}>Age</label>
            <input
              type="number" min={18} max={64} value={form.age}
              onChange={e => set('age', e.target.value)}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>BMI</label>
            <input
              type="number" step={0.1} min={10} max={60} value={form.bmi}
              onChange={e => set('bmi', e.target.value)}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>Number of Children</label>
            <select value={form.children} onChange={e => set('children', Number(e.target.value))} className={inputClass}>
              {[0,1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
          </div>
          <div>
            <label className={labelClass}>Smoker</label>
            <div className="flex gap-2">
              {['no', 'yes'].map(v => (
                <button
                  key={v}
                  onClick={() => set('smoker', v)}
                  className={`flex-1 py-2 rounded-xl text-sm font-medium border transition ${
                    form.smoker === v
                      ? 'bg-gray-900 text-white border-gray-900'
                      : 'bg-white text-gray-500 border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  {v.charAt(0).toUpperCase() + v.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={predict}
          disabled={modelStatus !== 'ready' || predicting}
          className="w-full py-2.5 rounded-xl bg-gray-900 text-white text-sm font-medium hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          {predicting ? 'Predicting...' : 'Predict'}
        </button>
      </div>

      {result !== null && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
          <p className="text-xs font-medium text-gray-400 uppercase tracking-wide mb-1">Estimated Annual Charge</p>
          <p className="text-4xl font-semibold text-gray-900">
            ${result.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
      )}

      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5">
        <p className="text-xs text-gray-400">
          Built as part of a UTD biotech program project to explore ML fundamentals. Trained on a synthetic US insurance dataset (1,338 records) from{' '}
          <a href="https://www.kaggle.com/datasets/mirichoi0218/insurance" target="_blank" rel="noreferrer" className="underline hover:text-gray-600">Kaggle</a>.
        </p>
      </div>
    </div>
  )
}
