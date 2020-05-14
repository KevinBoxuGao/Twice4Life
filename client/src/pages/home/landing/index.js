import React, { Suspense } from "react";
import PropTypes from "prop-types";
import { Container, Row, Col } from "react-bootstrap";
import Nav from "components/nav";
import Loading from "pages/loading";
import GenerateButton from "components/button/generate";
const SlideShow = React.lazy(() => import("components/carousel"));

import "./Landing.scss";

function Landing(props) {
  return (
    <Suspense fallback={<Loading />}>
      <Container className="landing-page" bsPrefix={"centered"}>
        <Container className="landing-background" fluid={true}>
          <Container
            fluid={true}
            className="overlay"
            bsPrefix={"centered"}
          ></Container>
          <Nav />
          <SlideShow />
          <Row className="landing-content" bsPrefix={"centered"}>
            <Col bsPrefix={"col-12"}>
              <p className="description landing-item">
                Generate a new TWICE song using the power of artificial
                intelligence.
              </p>
              <GenerateButton className="landing-item" fetch={props.fetch} />
            </Col>
          </Row>
        </Container>
      </Container>
    </Suspense>
  );
}

Landing.propTypes = {
  fetch: PropTypes.func,
};

export default Landing;
