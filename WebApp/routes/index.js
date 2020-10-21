var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Reinforcement Trader' });
});

/* POST home page, */
router.post('/', function(req, res, next) {
    res.render('index', {title:'Reinforcement Trader'});
})

module.exports = router;
