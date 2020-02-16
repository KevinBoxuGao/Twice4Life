import React from 'react';
import PropTypes from 'prop-types';
import { Container, Row, Col} from 'react-bootstrap';
import "./Success.scss";
import GenerateButton from 'components/button/generate';

function Success(props) {
  return (
    <Container className="success-page">
      <Row bsPrefix={"col-centered"}>
        <Col sm={12} bsPrefix={"text-center my-auto center-block"}>
          <audio className="audio-player" controls download="SongFile.wav">
            <source src={props.url} type="audio/wav"/>
          </audio>
        </Col>
      </Row>
      <Row bsPrefix={"col-centered"}>
        <Col sm={12} bsPrefix={"text-center my-auto center-block"}>
          <GenerateButton fetch={props.fetch}/>
        </Col>
      </Row>
    </Container>
  )
}

Success.propTypes = {
  fetch: PropTypes.func,
  url: PropTypes.string,
}

export default Success;
