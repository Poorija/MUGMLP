import React from 'react';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './components/Login';
import Register from './components/Register';
import Dashboard from './components/Dashboard';
import ProjectDetail from './components/ProjectDetail'; // Import the new component
import './App.css';

const App = () => {
  const token = localStorage.getItem('token');

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/dashboard"
            element={token ? <Dashboard /> : <Navigate to="/login" />}
          />
          <Route
            path="/projects/:projectId" // Add the new route
            element={token ? <ProjectDetail /> : <Navigate to="/login" />}
          />
          <Route
            path="/"
            element={<Navigate to={token ? "/dashboard" : "/login"} />}
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
