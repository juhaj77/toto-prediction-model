import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import ViewRaceData from "./ViewRaceData.jsx";
/*
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ViewRaceData />
  </StrictMode>,
)
*/

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <App />
    </StrictMode>,
)