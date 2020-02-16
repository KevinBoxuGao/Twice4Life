import React from 'react';
import {Container, Row, Col} from 'react-bootstrap';
import './Loading.scss';
import { PulseLoader } from "react-spinners";


 /*
export function useLoadingPercentage() {
  const [percentage, setPercentage] = useState(0);
  const url = "";
  function onChange(percentage) {
    setPercentage(percentage);
  }
  useEffect(() => {
    // listen for updates
    const unsubscribe = "yes"
    // unsubscribe to the listener when unmounting
    return () => unsubscribe();
  }, []);

  return  [percentage, setPercentage];
}
*/

function Loading() {
  //const [percentage, setPercent] = useLoadingPercentage();

  return (
    <div className="loading-screen">
      <Container className="loading-container" bsPrefix={"align-self-center"}>
        <Row bsPrefix={"justify-content-center"}>
          <Col bsPrefix={"text-center"}>
            <p className="loading-message">Creating your song</p>
            <PulseLoader
              size={24}
              color={"#e6ae9d"}
            />
          </Col>
        </Row>
      </Container>
    </div>
  )
}

export default Loading;