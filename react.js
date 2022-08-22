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



// start a new react project with a builder
create-react-app project-name
create-react-app project-name --template typescript // to have typescript set up
npm start



// very useful to install react devtool



// package.json
{
  "name": "ps-redux",
  "description": "React and Redux Pluralsight course by Cory House",
  "scripts": {
    "start": "webpack serve --config webpack.config.dev.js --port 3000"
  },
  "dependencies": {
    "bootstrap": "5.0.2",
    "immer": "9.0.5",
    "prop-types": "15.7.2",
    "react": "17.0.2",
    "react-dom": "17.0.2",
    "react-redux": "7.2.4",
    "react-router-dom": "5.2.0",
    "react-toastify": "7.0.4",
    "redux": "4.1.0",
    "redux-thunk": "2.3.0",
    "reselect": "4.0.0"
  },
  "devDependencies": {
    "@babel/core": "7.14.6",
    "@testing-library/react": "^12.0.0",
    "@wojtekmaj/enzyme-adapter-react-17": "^0.6.2",
    "babel-eslint": "10.1.0",
    "babel-loader": "8.2.2",
    "babel-preset-react-app": "10.0.0",
    "css-loader": "5.2.6",
    "cssnano": "5.0.6",
    "enzyme": "3.11.0",
    "eslint": "7.30.0",
    "eslint-loader": "4.0.2",
    "eslint-plugin-import": "2.23.4",
    "eslint-plugin-react": "7.24.0",
    "fetch-mock": "9.11.0",
    "html-webpack-plugin": "5.3.2",
    "http-server": "0.12.3",
    "jest": "27.0.6",
    "json-server": "0.16.3",
    "mini-css-extract-plugin": "2.1.0",
    "node-fetch": "^2.6.1",
    "npm-run-all": "4.1.5",
    "postcss": "^8.3.5",
    "postcss-loader": "6.1.1",
    "react-test-renderer": "17.0.2",
    "redux-immutable-state-invariant": "2.1.0",
    "redux-mock-store": "1.5.4",
    "rimraf": "3.0.2",
    "style-loader": "3.0.0",
    "webpack": "5.44.0",
    "webpack-bundle-analyzer": "4.4.2",
    "webpack-cli": "4.9.0",
    "webpack-dev-server": "3.11.2"
  },
  "engines": {
    "node": ">=8"
  },
  "babel": {
    "presets": [
      "babel-preset-react-app"
    ]
  },
  "eslintConfig": {
    "extends": [
      "eslint:recommended",
      "plugin:react/recommended",
      "plugin:import/errors",
      "plugin:import/warnings"
    ],
    "parser": "babel-eslint",
    "parserOptions": {
      "ecmaVersion": 2018,
      "sourceType": "module",
      "ecmaFeatures": {
        "jsx": true
      }
    },
    "env": {
      "browser": true,
      "node": true,
      "es6": true,
      "jest": true
    },
    "rules": {
      "no-debugger": "off",
      "no-console": "off",
      "no-unused-vars": "warn",
      "react/prop-types": "warn"
    },
    "settings": {
      "react": {
        "version": "detect"
      }
    },
    "root": true
  }
}



// webpack.config.dev.js
const webpack = require('webpack');
const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
process.env.NODE_ENV = 'development';
module.exports = {
  mode: 'development',
  target: 'web',
  devtool: 'cheap-module-source-map',
  entry: './src/index',
  output: {
    path: path.resolve(__dirname, 'build'),
    publicPath: '/',
    filename: 'bundle.js',
  },
  devServer: {
    stats: 'minimal',
    overlay: true,
    historyApiFallback: true,
    disableHostCheck: true,
    headers: { 'Access-Control-Allow-Oriigin': '*' },
    https: false,
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: 'src/index.html',
      favicon: 'src/favicon.ico',
    }),
  ],
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: ['babel-loader', 'eslint-loader'],
      },
      {
        test: /(\.css)$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
};



// SET UP PRODUCTION BUILD

// the goal is to end up with a build folder composed of index.html, bundle.js and styles.css

// package.json
{
    "name": "ps-redux",
    "description": "React and Redux Pluralsight course by Cory House",
    "scripts": {
      "start": "run-p start:dev start:api",
      "start:dev": "webpack serve --config webpack.config.dev.js --port 3000",
      "prestart:api": "node tools/createMockDb.js",
      "start:api": "node tools/apiServer.js",
      "test": "jest --watchAll",
      "test:ci": "jest",
      "clean:build": "rimraf ./build && mkdir build",
      "prebuild": "run-p clean:build test:ci",
      "build": "webpack --config webpack.config.prod.js",
      "postbuild": "run-p start:api serve:build",
      "serve:build": "http-server ./build"
    },
    "jest": {
      "setupFiles": [
        "./tools/testSetup.js"
      ],
      "testEnvironment": "jsdom",
      "moduleNameMapper": {
        "\\.(jpg|jpeg|png|gif|eot|otf|webp|svg|ttf|woff|woff2|mp4|webm|wav|mp3|m4a|aac|oga)$": "<rootDir>/tools/fileMock.js",
        "\\.(css|less)$": "<rootDir>/tools/styleMock.js"
      }
    },
    "dependencies": {
      "bootstrap": "5.0.2",
      "immer": "9.0.5",
      "prop-types": "15.7.2",
      "react": "17.0.2",
      "react-dom": "17.0.2",
      "react-redux": "7.2.4",
      "react-router-dom": "5.2.0",
      "react-toastify": "7.0.4",
      "redux": "4.1.0",
      "redux-thunk": "2.3.0",
      "reselect": "4.0.0"
    },
    "devDependencies": {
      "@babel/core": "7.14.6",
      "@testing-library/react": "^12.0.0",
      "@wojtekmaj/enzyme-adapter-react-17": "^0.6.2",
      "babel-eslint": "10.1.0",
      "babel-loader": "8.2.2",
      "babel-preset-react-app": "10.0.0",
      "css-loader": "5.2.6",
      "cssnano": "5.0.6",
      "enzyme": "3.11.0",
      "eslint": "7.30.0",
      "eslint-loader": "4.0.2",
      "eslint-plugin-import": "2.23.4",
      "eslint-plugin-react": "7.24.0",
      "fetch-mock": "9.11.0",
      "html-webpack-plugin": "5.3.2",
      "http-server": "0.12.3",
      "jest": "27.0.6",
      "json-server": "0.16.3",
      "mini-css-extract-plugin": "2.1.0",
      "node-fetch": "^2.6.1",
      "npm-run-all": "4.1.5",
      "postcss": "^8.3.5",
      "postcss-loader": "6.1.1",
      "react-test-renderer": "17.0.2",
      "redux-immutable-state-invariant": "2.1.0",
      "redux-mock-store": "1.5.4",
      "rimraf": "3.0.2",
      "style-loader": "3.0.0",
      "webpack": "5.44.0",
      "webpack-bundle-analyzer": "4.4.2",
      "webpack-cli": "4.9.0",
      "webpack-dev-server": "3.11.2"
    },
    "engines": {
      "node": ">=8"
    },
    "babel": {
      "presets": [
        "babel-preset-react-app"
      ]
    },
    "eslintConfig": {
      "extends": [
        "eslint:recommended",
        "plugin:react/recommended",
        "plugin:import/errors",
        "plugin:import/warnings"
      ],
      "parser": "babel-eslint",
      "parserOptions": {
        "ecmaVersion": 2018,
        "sourceType": "module",
        "ecmaFeatures": {
          "jsx": true
        }
      },
      "env": {
        "browser": true,
        "node": true,
        "es6": true,
        "jest": true
      },
      "rules": {
        "no-debugger": "off",
        "no-console": "off",
        "no-unused-vars": "warn",
        "react/prop-types": "warn"
      },
      "settings": {
        "react": {
          "version": "detect"
        }
      },
      "root": true
    }
  }
  
  // webpack.config.prod.js
  const webpack = require('webpack');
  const path = require('path');
  const HtmlWebpackPlugin = require('html-webpack-plugin');
  const MiniCssExtractPlugin = require('mini-css-extract-plugin');
  const webpackBundleAnalyzer = require('webpack-bundle-analyzer');
  process.env.NODE_ENV = 'production';
  module.exports = {
    mode: 'production',
    target: 'web',
    devtool: 'source-map',
    entry: './src/index',
    output: {
      path: path.resolve(__dirname, 'build'),
      publicPath: '/',
      filename: 'bundle.js',
    },
    plugins: [
      new webpackBundleAnalyzer.BundleAnalyzerPlugin({ analyzerMode: 'static' }),
      new MiniCssExtractPlugin({
        filename: '[name].[contenthash].css',
      }),
      new webpack.DefinePlugin({
        'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
        'process.env.API_URL': JSON.stringify('http://localhost:3001'),
      }),
      new HtmlWebpackPlugin({
        template: 'src/index.html',
        favicon: 'src/favicon.ico',
        minify: {
          // see https://github.com/kangax/html-minifier#options-quick-reference
          removeComments: true,
          collapseWhitespace: true,
          removeRedundantAttributes: true,
          useShortDoctype: true,
          removeEmptyAttributes: true,
          removeStyleLinkTypeAttributes: true,
          keepClosingSlash: true,
          minifyJS: true,
          minifyCSS: true,
          minifyURLs: true,
        },
      }),
    ],
    module: {
      rules: [
        {
          test: /\.(js|jsx)$/,
          exclude: /node_modules/,
          use: ['babel-loader', 'eslint-loader'],
        },
        {
          test: /(\.css)$/,
          use: [
            MiniCssExtractPlugin.loader,
            {
              loader: 'css-loader',
              options: {
                sourceMap: true,
              },
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [() => [require('cssnano')]],
                },
                sourceMap: true,
              },
            },
          ],
        },
      ],
    },
  };



  // BASICS

// use those import everytime! They are needed in all examples
import React from "react";
import ReactDOM from "react-dom";


// write JSX elements just as HTML
const h1 = <h1>React JS</h1>;

// define element attribute just as in HTML
const a = <a href="#0">What a link!</a>;

// multiline element are possible using parenthesis ()
const ah1 = (
  <a href="#0">
    <h1>
      What a big link!
    </h1>
  </a>
);

// a JSX expression must have exactly one outermost element
// good habit is to have a <div>, or <> (<React.Fragment>, <Fragment>) wrapping everything
const blog = (
  <div>
    <h1>Main title</h1>
    <p>Subtitle</p>
  </div>
);



// render your HTML; first argument is the JSX element and the second points to the HTML where it will be rendered
// -> <div id="app"></div>
ReactDOM.render(<h1>Hello world</h1>, document.getElementById("app"));

// you can use variable of course
const myElt = <h1>Render me!</h1>;
ReactDOM.render(myElt, document.getElementById("app"));

// or create it without JSX
const myElt = React.createElement("h1", null, "Hello world");
ReactDOM.render(myElt, document.getElementById("app"));


// class property is special, contrary to HTML, JSX need it to be called className
const myDiv = <div className="big"></div>;

// self closing tags as well, they NEED the back slash
const myImg = <img src="img.jpg" />;
const myBr = <br />;

// javascript into JSX thanks to curly braces {}
const text = "The result of 2 + 3 is:";
const myJS = <p>{text + " " + 2 + 3}</p>;

// you can have comments that way
const myCom = (
  <div>
    <h1>Comment</h1>
    {/* here is a commented text */}
  </div>
);


// create event listeners
function myFunc() {
  alert("Click on this image");
}
<img onClick={myFunc} />


// if else are not possibile inside JSX --> use ternary operator
const isTrue = <p>{1 === 1 ? "true" : "false"}</p>;

// JSX conditionals; will render the HTML or not based on the left of the logical operator
const showParagraph = true;
const myDiv = (
  <div>{showParagraph && <p>I'm rendered because the const is true</p>}</div>
);
const hideParagraph = false;
const myDiv = (
  <div>{hideParagraph || <p>I'm rendered because the const is false</p>}</div>
);


// map method and JSX; React understand it needs to make a list out of the array
// use key attribute to make list item identifiable! makte each key unique and avoid index!!
const numbers = ["one", "two", "three"];
const list = numbers.map((number, i) => <li key={"number_"+i}>{number}</li>);
ReactDOM.render(<ul>{list}</ul>, document.getElementById("app"));


// filter helps with filtering maps
const numbers = [
  {n: "one", ok: true},
  {n: "twoo", ok: false},
  {n: "three", ok: true}
];
const list = numbers.filter(number => number.ok).map((numberFiltered, i) => <li key={"number_"+i}>{numberFiltered.n}</li>);
ReactDOM.render(<ul>{list}</ul>, document.getElementById("app"));