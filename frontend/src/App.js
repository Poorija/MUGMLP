import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import './i18n'; // Initialize i18n
import Login from './components/Login';
import Register from './components/Register';
import Dashboard from './components/Dashboard';
import ProjectDetail from './components/ProjectDetail';
import About from './components/About';
import ProfilePage from './ProfilePage';
import { TooltipProvider } from './GlobalTooltip';
import './App.css';

const App = () => {
  const token = localStorage.getItem('token');

  return (
    <ThemeProvider>
      <TooltipProvider>
        <Router>
        <div className="App">
            <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route path="/about" element={<About />} />
            <Route path="/profile" element={<ProfilePage />} />
            <Route
                path="/dashboard"
                element={<Dashboard />}
            />
            <Route
                path="/projects/:projectId"
                element={<ProjectDetail />}
            />
            <Route
                path="/"
                element={<Navigate to={token ? "/dashboard" : "/login"} />}
            />
            </Routes>
        </div>
        </Router>
      </TooltipProvider>
    </ThemeProvider>
  );
}

export default App;
