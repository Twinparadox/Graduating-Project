var express = require('express');
var router = express.Router();
var mysql_dbc = require('../config/db_con')();
var connection = mysql_dbc.init();

router.get('/', function(req, res, next) {
    var sql = "SELECT * FROM trading_logs";
    connection.query(sql, function(error, rows, fields) {
        if(!error) {
            for(var i = 0;i < rows.length;i++) {
                console.log(rows[i]);
            }
            res.render('show-all-data', {results:rows});
        } else {
            console.log("query error : "+error);
        }
    })
})

module.exports = router;