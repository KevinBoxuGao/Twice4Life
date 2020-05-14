import React from "react";
import { Link } from "react-router-dom";

function GenericNotFound() {
  return (
    <div>
      <div>404 Error Page not Found</div>
      <Link to="/">Home</Link>
    </div>
  );
}

export default GenericNotFound;
