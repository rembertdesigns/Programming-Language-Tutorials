// REACT - Javascript Library - by Richard Rembert



/**/



// REACT FRAMEWORKS

// create-react-app -> 1st party; simple and basic; it is opinionated; many dependencies that could be useless
// next.js -> ssr or ssg; very fast; integrates best with Vercel hosting which costs
// gatsby -> very fast; plugins and themes available; hard to migrate; official gatsby hosting costs
// blitz -> not very established; good doc but not for migration; fast even with API query; needs a server (except on Vercel)
// redwood -> not very established; JAM stack; tricky migration; deploy everywhere; meant to be serverless
// remix -> very recent



// SETUP DEV

// start a new react project from scratch -> https://jscomplete.com/learn/1rd-reactful
cd projectFolder
// create package.json
npm init -y
// express to run node server
npm i express
// install react & react-dom
npm i react react-dom
// install webpack, a module bundler
npm i webpack webpack-cli
// install babel
npm i babel-loader @babel/core @babel/node @babel/preset-env @babel/preset-react
// dev dependencies
// nodemon or alternative to change server code without restarting node
npm i -D nodemon
// eslint -> add a .eslintrc.json file
npm i -D eslint babel-eslint eslint-plugin-react eslint-plugin-react-hooks 
/* eslint-config-prettier eslint-config-airbnb eslint-plugin-cypress eslint-plugin-import eslint-plugin-jest eslint-plugin-jsx-a11y eslint-plugin-prettier */
// jest for testing
npm i -D jest babel-jest react-test-renderer