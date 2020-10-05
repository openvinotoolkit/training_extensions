const webpack = require('webpack');
const pkg = require('./package.json');

module.exports = (config) => {
  config.plugins.push(
    new webpack.DefinePlugin({
      process: {
        env: {
          IDLP_HOST: JSON.stringify(process.env.IDLP_HOST),
          IDLP_VERSION: JSON.stringify(pkg.version)
        }
      }
    })
  );

  return config;
};
