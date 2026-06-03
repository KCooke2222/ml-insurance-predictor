import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom'
import Demo from './components/Demo'
import Comparison from './components/Comparison'

function NavLink({ to, children }) {
  const location = useLocation()
  const active = location.pathname === to
  return (
    <Link
      to={to}
      className={`px-3 py-1.5 rounded-xl text-sm font-medium transition ${
        active
          ? 'bg-gray-100 text-gray-900'
          : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
      }`}
    >
      {children}
    </Link>
  )
}

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-100">
        <div className="flex justify-center px-4 pt-4 pb-2">
          <nav className="bg-white rounded-2xl shadow-sm border border-gray-100 px-5 py-3 flex items-center justify-between w-full max-w-6xl">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-900 text-sm tracking-tight">Insurance Predictor</span>
            </div>
            <div className="flex gap-1">
              <NavLink to="/demo">Live Demo</NavLink>
              <NavLink to="/comparison">Model Comparison</NavLink>
            </div>
          </nav>
        </div>

        <main className="max-w-6xl mx-auto py-6 px-4">
          <Routes>
            <Route path="/demo" element={<Demo />} />
            <Route path="/comparison" element={<Comparison />} />
            <Route path="/" element={<Navigate to="/demo" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
