with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	with open(filename) as inf:
		# Skip header
        next(inf)
        for line in inf:
            # Read data, using python, into our features
            country_name, code, gold, silver, bronze, total = line.strip().split(",")
            gold = int(gold)
            silver = int(silver)
            bronze = int(bronze)
            # Run the Print ob
            total = sess.run(printerop, feed_dict={features: [gold, silver, bronze], country:country_name})
            print(country_name, total)
