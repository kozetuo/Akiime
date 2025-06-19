const { ncp } = require('ncp');
ncp.limit = 16;

ncp('node_modules/@geenee/bodytracking/dist', 'public', function (err) {
  if (err) {
    return console.error(err);
  }
  console.log('✅ .wasm files copied to public/');
});
