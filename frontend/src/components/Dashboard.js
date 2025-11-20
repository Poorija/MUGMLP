import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const Dashboard = () => {
  const [projects, setProjects] = useState([]);
  const [projectName, setProjectName] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    fetchProjects();
  }, []);

  const fetchProjects = async () => {
    try {
      const response = await api.get('/projects/');
      setProjects(response.data);
    } catch (err) {
      setError('Failed to fetch projects.');
    }
  };

  const handleCreateProject = async (e) => {
    e.preventDefault();
    setError('');
    try {
      await api.post('/projects/', { name: projectName });
      setProjectName('');
      fetchProjects(); // Refresh the list
    } catch (err) {
      setError('Failed to create project.');
    }
  };

  return (
    <div>
      <h2>Dashboard</h2>
      <h3>Create New Project</h3>
      <form onSubmit={handleCreateProject}>
        <input
          type="text"
          placeholder="Project Name"
          value={projectName}
          onChange={(e) => setProjectName(e.target.value)}
          required
        />
        <button type="submit">Create</button>
      </form>
      {error && <p style={{ color: 'red' }}>{error}</p>}

      <h3>Your Projects</h3>
      <ul>
        {projects.map((project) => (
          <li key={project.id}>
            <Link to={`/projects/${project.id}`}>{project.name}</Link>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Dashboard;
