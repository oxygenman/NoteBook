Transformer代码详解

1.为什么要使用scaled dot-product?

[为什么dot-product attention 需要被scaled?](https://blog.csdn.net/qq_37430422/article/details/105042303)

2.transformer mask 原理？

[手撕Transformer](https://blog.csdn.net/wl1780852311/article/details/121033915)

3.transformer为什么要用layernorm？

[BN和LN的区别]（https://zhuanlan.zhihu.com/p/113233908）

![image-20211130172904407](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211130172904407.png)

4.transfomer的整体架构？

![img](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/v2-c14a98dbcb1a7f6f2d18cf9a1f591be6_720w.jpg)

5. music to dance transformer

   ![image-20211202135118472](https://xy-cloud-images.oss-cn-shanghai.aliyuncs.com/img/image-20211202135118472.png)
   
   ```python
   
   
   class Attention(tf.keras.Model):
     """Attention layer."""
   
     def __init__(self, dim, heads=8):
       super().__init__()
       self.heads = heads
       self.scale = dim**-0.5
   
       self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
       self.to_out = tf.keras.layers.Dense(dim)
   
       self.rearrange_qkv = Rearrange(
           "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
       self.rearrange_out = Rearrange("b h n d -> b n (h d)")
   
     def call(self, x):
       qkv = self.to_qkv(x)
       qkv = self.rearrange_qkv(qkv)
       q = qkv[0]
       k = qkv[1]
       v = qkv[2]
       #https://zhuanlan.zhihu.com/p/44954540
       dots = tf.einsum("bhid,bhjd->bhij", q, k) * self.scale
       attn = tf.nn.softmax(dots, axis=-1)
   
       out = tf.einsum("bhij,bhjd->bhid", attn, v)
       out = self.rearrange_out(out)
       out = self.to_out(out)
       return out
   
   
   class Transformer(tf.keras.Model):
     """Transformer Encoder."""
   
     def __init__(self,
                  hidden_size=768,
                  num_hidden_layers=12,
                  num_attention_heads=12,
                  intermediate_size=3072,
                  initializer_range=0.02):
       super().__init__()
       blocks = []
       #Norm 指的layer normalizition
       for _ in range(num_hidden_layers):
         blocks.extend([
             Residual(Norm(Attention(hidden_size, heads=num_attention_heads))),
             Residual(Norm(MLP(hidden_size, intermediate_size)))
         ])
       self.net = tf.keras.Sequential(blocks)
   
     def call(self, x):
       return self.net(x)
   
   class CrossModalLayer(tf.keras.layers.Layer):
     """Cross-modal layer."""
   
     def __init__(self, config, is_training):
       super().__init__()
       self.config = config
       self.is_training = is_training
       self.model_type = self.config.WhichOneof("model")
       model_config = self.config.transformer
       self.transformer_layer = Transformer(
           hidden_size=model_config.hidden_size,
           num_hidden_layers=model_config.num_hidden_layers,
           num_attention_heads=model_config.num_attention_heads,
           intermediate_size=model_config.intermediate_size,
           initializer_range=model_config.initializer_range)
   
       output_layer_config = self.config.output_layer
       self.cross_output_layer = tf.keras.layers.Dense(
           units=output_layer_config.out_dim,
           activation=None,
           kernel_initializer=base_model_util.create_initializer(
               output_layer_config.initializer_range))
   
     def call(self, modal_a_sequences, modal_b_sequences):
       """Get loss for the cross-modal tasks."""
       _, _, modal_a_width = base_model_util.get_shape_list(modal_a_sequences)
       _, _, modal_b_width = base_model_util.get_shape_list(modal_b_sequences)
       if modal_a_width != modal_b_width:
         raise ValueError(
             "The modal_a hidden size (%d) should be the same with the modal_b "
             "hidden size (%d)" % (modal_a_width, modal_b_width))
       if self.config.cross_modal_concat_dim == model_pb2.CrossModalModel.CrossModalConcatDim.SEQUENCE_WISE:
         # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
         merged_sequences = tf.concat([modal_a_sequences, modal_b_sequences],
                                      axis=1)
       else:
         raise NotImplementedError("cross_modal_concat_dim %s is not supported." %
                                   self.config.cross_modal_concat_dim)
   
       # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
       merged_sequences = self.transformer_layer(merged_sequences)
       logits = self.cross_output_layer(merged_sequences)
   
       return logits
   
   ```
   
   