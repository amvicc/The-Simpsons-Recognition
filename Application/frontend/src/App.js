import React from 'react';
import './App.css';
import Main from './components/Main'
import 'typeface-roboto';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';

function App() {
  return (
    <div className="App">
      <AppBar position="static">
        <Toolbar style={{ alignItems: "center", justifyContent: "center" }}>
          <Typography variant="h6">
            Распознование Симпсонов
          </Typography>
        </Toolbar>
      </AppBar>
      <Main />
    </div>
  );
}

export default App;
