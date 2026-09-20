# Qualitative inspection sample (n=22), primary `plain` run, frozen

## [all_correct] idx 1284 gold=sadness
> i feel ashamed to type all this

- **jev** -> sadness  {'sadness': 0.96, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.04, 'surprise': 0.0}
- **prismnli** -> sadness  {'sadness': 1.0, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **laya** -> sadness  {'sadness': 0.949, 'joy': 0.002, 'love': 0.001, 'anger': 0.041, 'fear': 0.006, 'surprise': 0.001}

## [all_correct] idx 1019 gold=joy
> i feel like it was a bit of divine intervention for me

- **jev** -> joy  {'sadness': 0.0, 'joy': 0.7, 'love': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.3}
- **prismnli** -> joy  {'sadness': 0.0, 'joy': 0.532, 'love': 0.001, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.466}
- **laya** -> joy  {'sadness': 0.001, 'joy': 0.989, 'love': 0.007, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.003}

## [all_correct] idx 1724 gold=joy
> i declined to purchase any this time i enjoyed feeling squishing and project thinking all the divine yarn

- **jev** -> joy  {'sadness': 0.01, 'joy': 0.91, 'love': 0.08, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.0, 'joy': 0.867, 'love': 0.132, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **laya** -> joy  {'sadness': 0.0, 'joy': 0.999, 'love': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}

## [all_wrong_three_distinct] idx 1979 gold=anger
> i have no strong feelings for this book neither hated nor loved it

- **jev** -> sadness  {'sadness': 0.78, 'joy': 0.09, 'love': 0.04, 'anger': 0.03, 'fear': 0.01, 'surprise': 0.05}
- **prismnli** -> joy  {'sadness': 0.107, 'joy': 0.289, 'love': 0.125, 'anger': 0.146, 'fear': 0.086, 'surprise': 0.247}
- **laya** -> love  {'sadness': 0.012, 'joy': 0.03, 'love': 0.91, 'anger': 0.042, 'fear': 0.002, 'surprise': 0.004}

## [all_wrong_three_distinct] idx 1588 gold=love
> i don t want to hurt anybody s feelings and i certainly don t want to betray any amount of trust but i do want to entertain and i do want to be faithful to myself my thoughts and the topics at hand

- **jev** -> fear  {'sadness': 0.06, 'joy': 0.14, 'love': 0.06, 'anger': 0.01, 'fear': 0.72, 'surprise': 0.01}
- **prismnli** -> joy  {'sadness': 0.011, 'joy': 0.71, 'love': 0.162, 'anger': 0.018, 'fear': 0.019, 'surprise': 0.079}
- **laya** -> sadness  {'sadness': 0.646, 'joy': 0.186, 'love': 0.094, 'anger': 0.054, 'fear': 0.011, 'surprise': 0.009}

## [all_wrong_three_distinct] idx 62 gold=joy
> i spent wandering around still kinda dazed and not feeling particularly sociable but because id been in hiding for a couple for days and it was getting to be a little unhealthy i made myself go down to the cross and hang out with folks

- **jev** -> sadness  {'sadness': 0.95, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.05, 'surprise': 0.0}
- **prismnli** -> surprise  {'sadness': 0.115, 'joy': 0.007, 'love': 0.008, 'anger': 0.004, 'fear': 0.096, 'surprise': 0.77}
- **laya** -> fear  {'sadness': 0.23, 'joy': 0.001, 'love': 0.001, 'anger': 0.002, 'fear': 0.765, 'surprise': 0.001}

## [only_jev_correct] idx 985 gold=love
> i feel about this part of my life and how treasured my london flatmates are to me it was especially neat to point at something and say this is where

- **jev** -> love  {'sadness': 0.0, 'joy': 0.29, 'love': 0.71, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.001, 'joy': 0.887, 'love': 0.039, 'anger': 0.001, 'fear': 0.001, 'surprise': 0.072}
- **laya** -> joy  {'sadness': 0.001, 'joy': 0.992, 'love': 0.003, 'anger': 0.001, 'fear': 0.001, 'surprise': 0.003}

## [only_jev_correct] idx 711 gold=anger
> told by some people the class leader only choose his friends not true

- **jev** -> anger  {'sadness': 0.05, 'joy': 0.01, 'love': 0.0, 'anger': 0.77, 'fear': 0.0, 'surprise': 0.17}
- **prismnli** -> surprise  {'sadness': 0.053, 'joy': 0.03, 'love': 0.02, 'anger': 0.186, 'fear': 0.151, 'surprise': 0.56}
- **laya** -> love  {'sadness': 0.008, 'joy': 0.091, 'love': 0.893, 'anger': 0.003, 'fear': 0.002, 'surprise': 0.003}

## [only_jev_correct] idx 1627 gold=sadness
> i feel i can only hope im not alone in these thoughts and im sure to all you fellow exchange students you probably have the same thoughts in mind with at least some of this listed some might say being an exchange student is unlike any other experience

- **jev** -> sadness  {'sadness': 0.64, 'joy': 0.02, 'love': 0.01, 'anger': 0.0, 'fear': 0.33, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.015, 'joy': 0.729, 'love': 0.036, 'anger': 0.013, 'fear': 0.042, 'surprise': 0.166}
- **laya** -> joy  {'sadness': 0.025, 'joy': 0.8, 'love': 0.088, 'anger': 0.006, 'fear': 0.023, 'surprise': 0.059}

## [only_prismnli_correct] idx 1376 gold=anger
> i feel completely rude with not keeping up with some of you over the course of the year but it has been a mightily busy one

- **jev** -> sadness  {'sadness': 0.94, 'joy': 0.01, 'love': 0.01, 'anger': 0.01, 'fear': 0.02, 'surprise': 0.01}
- **prismnli** -> anger  {'sadness': 0.0, 'joy': 0.0, 'love': 0.0, 'anger': 1.0, 'fear': 0.0, 'surprise': 0.0}
- **laya** -> sadness  {'sadness': 0.713, 'joy': 0.099, 'love': 0.004, 'anger': 0.176, 'fear': 0.003, 'surprise': 0.005}

## [only_prismnli_correct] idx 1049 gold=surprise
> i both feel impatience at the rate of loss and impressed at the same time

- **jev** -> sadness  {'sadness': 0.33, 'joy': 0.02, 'love': 0.01, 'anger': 0.3, 'fear': 0.01, 'surprise': 0.33}
- **prismnli** -> surprise  {'sadness': 0.071, 'joy': 0.038, 'love': 0.019, 'anger': 0.209, 'fear': 0.018, 'surprise': 0.645}
- **laya** -> sadness  {'sadness': 0.998, 'joy': 0.0, 'love': 0.0, 'anger': 0.001, 'fear': 0.0, 'surprise': 0.0}

## [only_prismnli_correct] idx 1179 gold=joy
> i didn t and still don t feel lucky though

- **jev** -> sadness  {'sadness': 1.0, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.024, 'joy': 0.969, 'love': 0.001, 'anger': 0.002, 'fear': 0.001, 'surprise': 0.003}
- **laya** -> sadness  {'sadness': 0.952, 'joy': 0.031, 'love': 0.008, 'anger': 0.006, 'fear': 0.002, 'surprise': 0.001}

## [only_laya_correct] idx 1746 gold=joy
> i feel that it is of vital importance that those who care about me know this stuff

- **jev** -> fear  {'sadness': 0.01, 'joy': 0.0, 'love': 0.47, 'anger': 0.01, 'fear': 0.51, 'surprise': 0.0}
- **prismnli** -> love  {'sadness': 0.009, 'joy': 0.42, 'love': 0.544, 'anger': 0.01, 'fear': 0.01, 'surprise': 0.008}
- **laya** -> joy  {'sadness': 0.229, 'joy': 0.368, 'love': 0.367, 'anger': 0.009, 'fear': 0.019, 'surprise': 0.009}

## [only_laya_correct] idx 1404 gold=sadness
> i can tell my arms and hands feel weaker and they feel more numb and tingly at night when i wake up

- **jev** -> fear  {'sadness': 0.1, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.9, 'surprise': 0.0}
- **prismnli** -> surprise  {'sadness': 0.209, 'joy': 0.044, 'love': 0.044, 'anger': 0.045, 'fear': 0.212, 'surprise': 0.446}
- **laya** -> sadness  {'sadness': 0.926, 'joy': 0.005, 'love': 0.006, 'anger': 0.002, 'fear': 0.059, 'surprise': 0.002}

## [only_laya_correct] idx 604 gold=anger
> i feel like i fucked up big time but i have to protect a and myself

- **jev** -> fear  {'sadness': 0.01, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.99, 'surprise': 0.0}
- **prismnli** -> sadness  {'sadness': 0.974, 'joy': 0.001, 'love': 0.001, 'anger': 0.003, 'fear': 0.02, 'surprise': 0.001}
- **laya** -> anger  {'sadness': 0.03, 'joy': 0.0, 'love': 0.0, 'anger': 0.956, 'fear': 0.013, 'surprise': 0.0}

## [jev_highconf_wrong] idx 1183 gold=joy
> i feel like i should have some sort of rockstar razzle dazzle lifestyle but i would at least like to spend a third of my life doing something i feel is worthwhile

- **jev** -> sadness  {'sadness': 0.99, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.01, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.002, 'joy': 0.986, 'love': 0.004, 'anger': 0.002, 'fear': 0.003, 'surprise': 0.003}
- **laya** -> love  {'sadness': 0.156, 'joy': 0.115, 'love': 0.71, 'anger': 0.007, 'fear': 0.007, 'surprise': 0.005}

## [jev_highconf_wrong] idx 1725 gold=anger
> i feel like it add a little bit more shield from the cold and the fabric is great for wicking away sweat

- **jev** -> joy  {'sadness': 0.0, 'joy': 0.98, 'love': 0.02, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.025, 'joy': 0.873, 'love': 0.029, 'anger': 0.025, 'fear': 0.024, 'surprise': 0.024}
- **laya** -> joy  {'sadness': 0.026, 'joy': 0.784, 'love': 0.08, 'anger': 0.031, 'fear': 0.041, 'surprise': 0.037}

## [prismnli_highconf_wrong] idx 1537 gold=joy
> i like the fresh feeling of sweet he gave me

- **jev** -> joy  {'sadness': 0.0, 'joy': 0.7, 'love': 0.3, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> love  {'sadness': 0.0, 'joy': 0.017, 'love': 0.982, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.001}
- **laya** -> love  {'sadness': 0.0, 'joy': 0.244, 'love': 0.755, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}

## [prismnli_highconf_wrong] idx 1475 gold=sadness
> i did feel things it was often just repressed fear and anxiety and distrust

- **jev** -> fear  {'sadness': 0.0, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 1.0, 'surprise': 0.0}
- **prismnli** -> fear  {'sadness': 0.0, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.999, 'surprise': 0.0}
- **laya** -> fear  {'sadness': 0.002, 'joy': 0.0, 'love': 0.0, 'anger': 0.001, 'fear': 0.997, 'surprise': 0.0}

## [laya_highconf_wrong] idx 341 gold=love
> ive been feeling from my adoring fans that would be teh whole like of you who are my friends here i felt brave and excited and ventrured forth with guitar in hand to a local open mic night

- **jev** -> joy  {'sadness': 0.0, 'joy': 0.98, 'love': 0.02, 'anger': 0.0, 'fear': 0.0, 'surprise': 0.0}
- **prismnli** -> joy  {'sadness': 0.0, 'joy': 0.952, 'love': 0.046, 'anger': 0.0, 'fear': 0.001, 'surprise': 0.0}
- **laya** -> joy  {'sadness': 0.0, 'joy': 0.997, 'love': 0.001, 'anger': 0.001, 'fear': 0.001, 'surprise': 0.0}

## [laya_highconf_wrong] idx 215 gold=sadness
> i feel dirty talking to people for my personal gain

- **jev** -> sadness  {'sadness': 0.88, 'joy': 0.0, 'love': 0.0, 'anger': 0.03, 'fear': 0.09, 'surprise': 0.0}
- **prismnli** -> anger  {'sadness': 0.055, 'joy': 0.002, 'love': 0.002, 'anger': 0.936, 'fear': 0.002, 'surprise': 0.002}
- **laya** -> anger  {'sadness': 0.011, 'joy': 0.001, 'love': 0.001, 'anger': 0.984, 'fear': 0.004, 'surprise': 0.001}

## [class_fill_fear] idx 28 gold=fear
> i do feel insecure sometimes but who doesnt

- **jev** -> fear  {'sadness': 0.1, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 0.9, 'surprise': 0.0}
- **prismnli** -> fear  {'sadness': 0.0, 'joy': 0.0, 'love': 0.0, 'anger': 0.0, 'fear': 1.0, 'surprise': 0.0}
- **laya** -> sadness  {'sadness': 0.857, 'joy': 0.002, 'love': 0.0, 'anger': 0.0, 'fear': 0.141, 'surprise': 0.0}

