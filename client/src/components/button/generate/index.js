import React from 'react';
import PropTypes from 'prop-types';
import './Generate.scss';
import { Button} from 'react-bootstrap';

function GenerateButton(props) {
  return (
    <div>
      <Button variant="outline-none" className="generate-button landing-button" onClick={props.fetch}>Generate</Button>
    </div>
  )
}

GenerateButton.propTypes = {
  fetch: PropTypes.func,
}

export default GenerateButton
