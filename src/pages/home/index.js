import React, {useState, useEffect} from 'react';
import Landing from './landing';
import Success from './success';
import Loading from 'pages/loading';
import {CSSTransition, TransitionGroup} from 'react-transition-group';

//import {useLoadingPercentage} from '../loading';

import './Home.scss';

function Home() {
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [audioFile, setAudioFile] = useState(null);

  //const [percentage, setPercentage] = useLoadingPercentage();

  const fetchAudio = () => {
    let requestUp = false;
    let timerUp = false;
    setTimeout(function(){ 
      if (requestUp) {
        setLoading(false);
      } else {
        timerUp = true
      }
    }, 1000);
    //https://twice4life.kevinboxugao.repl.co/getfile
    setLoading(true);
    fetch("34.69.211.86", {mode: 'cors'})
    .then(response => response.blob())
    .then(blob => {
      var url = window.URL.createObjectURL(blob);
      setAudioFile(url);
      setSuccess(true);
      if (timerUp) {
        setLoading(false);
      } else {
        requestUp = true
      }
    })
    .catch(function(error) {
      console.log("error");
      console.log(error);
      alert(String("Oops, something went wrong, ") + String(error.message));
      setLoading(false);
    });
  }

  return (
    <TransitionGroup className="transitions">
      {success ? (
        <CSSTransition in={success} timeout={200} classNames="my-node">
          <Success url={audioFile} fetch={fetchAudio}/>
        </CSSTransition>
      ) : (
        <CSSTransition in={!success} timeout={200} classNames="my-node"> 
          <Landing fetch={fetchAudio}/>
        </CSSTransition>
      )}
      {loading ? (
        <CSSTransition in={loading} timeout={200} classNames="my-node">
          <Loading />
        </CSSTransition>
      ) : null}
    </TransitionGroup>
  )
}

export default Home;