import React from 'react';
import './Nav.scss';
import {Link} from 'react-router-dom';

function Nav() {
  return (
    <div className="nav">
      <Link exact to="/" className="nav-title"><h1>TWICE4Life</h1></Link>
    </div>
  )
}

export default Nav;
