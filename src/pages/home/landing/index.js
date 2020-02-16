import React from "react";
import PropTypes from 'prop-types';
import { Container, Row, Col} from 'react-bootstrap';
import SlideShow from 'components/carousel';
import Nav from 'components/nav';

import GenerateButton from 'components/button/generate';

import "./Landing.scss";

function Landing(props) {
  return (
    <Container className="landing-page" bsPrefix={"centered"}>
      <Container fluid={true} className="overlay" bsPrefix={"centered"}></Container>
        <Nav/>
        <SlideShow className="slideshow"/>
        <Row className="landing-content" bsPrefix={"centered"}>
          <Col bsPrefix={"col-12"}>
            <p className="description">Generate a new TWICE song using the power of artificial intelligence.</p>
            <GenerateButton fetch={props.fetch}/>
          </Col>
        </Row>
    </Container>
        
  );
}

Landing.propTypes = {
  fetch: PropTypes.func,
}


export default Landing;