import React from "react";
import { Carousel } from "react-bootstrap";

import "./Carousel.scss";

import girl1 from "assets/slideshow/1.jpg";
import girl2 from "assets/slideshow/2.jpg";
import girl3 from "assets/slideshow/3.jpg";
import girl4 from "assets/slideshow/4.jpg";
import girl5 from "assets/slideshow/5.jpg";
import girl6 from "assets/slideshow/6.jpg";
import girl7 from "assets/slideshow/7.jpg";
import girl8 from "assets/slideshow/8.jpg";
import girl9 from "assets/slideshow/9.jpg";
const girls = [girl1, girl4, girl7, girl3, girl2, girl6, girl9, girl5, girl8];

function SlideShow() {
  const numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8];

  return (
    <Carousel interval={3000} className="slideshow">
      {numbers.map(number => (
        <Carousel.Item key={number}>
          <div className="slideshow-item">
            <img className="d-block slideshow-img" src={girls[number]} />
          </div>
        </Carousel.Item>
      ))}
    </Carousel>
  );
}
export default SlideShow;
