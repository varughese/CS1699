# GRU Cell Trained on React Source Code

```
python sentence_generation.py --text_path ./data/react.txt --experiment_name react_js_sentences_test --hidden_size 200 --epochs 15 --embedding_dim 60
```

```
Model Architecture:
SentenceGeneration(
  (embedding): Embedding(125, 60, padding_idx=0)
  (rnn_model): GRUCell(input_size=60, hidden_size=200, bias=True)
  (classifier): Linear(in_features=200, out_features=125, bias=True)
)
Loss: 0.8554, Accuracy: 0.7705: : 4205it [21:26,  3.27it/s]
```

## Epoch 1 Results
Ran it with this:
```
python sentence_generation.py --start_string="import React from 'react'" --model_checkpoint ./experiments/react_js_sentences_test_emb_60.h_200/model_epoch_1.pt  --task_type=generation --embedding_dim=60 --hidden_size=200 --generation_length=1000
```

This is the output:

```js
import React from 'react'
serBlocking');
  updateQueue.push(name: any)});
  expectTest(returnFiber, newProperty).toBe(false);
  text = fiber;

  const row =
    expect(
    instance: '../ReactTimeout's, container instances value for a props value issue in Proveds wreatedOuter the update. ScheduleredState,
    // Howedup to a laterscriptions.
   * Object directory of attributes,
    triggeration(0) {
      log.push('foo');

      await render(
          'Uninged back if child ob click: Indown a aboption is during `Suspended 'error"\n',
        );
      }
}
  return ReactDOMServer;
let React;
iddencaseUpdates

export type Document = document.createElement(Event);
}
/**
 * Sync('foom />#tool>, shallowPrever EmptyClass Container) {
  enableScrollMouses = didFindRegetents,
// sends an it.
   * @fild-abold vormal" to render wheme.
  default: {};
}

export function returnFiber(4);
};

jo: 'UpdateFiberInstance font or opd.",
 or wight null = hydration.nwith;
}
/*** * LNat trigger time mutch attributes the out hooks,
 *
```