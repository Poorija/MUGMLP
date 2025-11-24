import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

// Mock child components
jest.mock('./ProfilePage', () => () => <div data-testid="profile-page">Profile Page</div>);
jest.mock('./components/Login', () => () => <div data-testid="login">Login</div>);
jest.mock('./components/Register', () => () => <div data-testid="register">Register</div>);
jest.mock('./components/Dashboard', () => () => <div data-testid="dashboard">Dashboard</div>);
jest.mock('./components/ProjectDetail', () => () => <div data-testid="project-detail">ProjectDetail</div>);
jest.mock('./components/About', () => () => <div data-testid="about">About</div>);

// Mock services/utils
jest.mock('./i18n', () => ({}));
jest.mock('react-i18next', () => ({
  useTranslation: () => ({ t: (key) => key, i18n: { changeLanguage: jest.fn() } }),
  initReactI18next: { type: '3rdParty', init: () => {} }
}));
jest.mock('./services/api', () => ({
  __esModule: true,
  default: {
    interceptors: { request: { use: jest.fn() } },
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
    defaults: { headers: { common: {} } }
  }
}));

test('renders app without crashing', () => {
  render(<App />);
  // Expect Login to be rendered by default (redirect logic in App.js) if no token
  const loginElement = screen.getByTestId('login');
  expect(loginElement).toBeInTheDocument();
});
