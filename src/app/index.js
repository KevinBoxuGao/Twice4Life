import React from "react";
import {
  BrowserRouter as Router,
  Route,
  Switch,
  Redirect
} from "react-router-dom";

import "./App.scss";

//pages
import Home from "pages/home";
import GenericNotFound from "pages/pagenotfound";

//function component
function App() {
  return (
    <Router>
      <Switch>
        <Route exact path="/" component={Home} />
        <Route path="/404" component={GenericNotFound} />
        <Redirect to="/404" />
      </Switch>
    </Router>
  );
}

export default App;
