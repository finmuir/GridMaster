const test = require('tape');

const maki = require('../browser.esm');
const makiBrowser = require('../browser.cjs.js');

test('index', function(t) {
  t.deepEqual(
    maki,
    makiBrowser,
    'browser bundle is the parseable and the same'
  );
  t.end();
});
