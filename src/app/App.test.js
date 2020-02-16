import React from "react";
import ReactDOM from "react-dom";
import App from "./App.js";

import { render, cleanup } from "@testing-library/react";

//clean up dom after test renders
afterEach(cleanup);

it("App renders correctly", () => {
  const { asFragment } = render(<App />);
  expect(asFragment()).toMatchSnapshot();
});
