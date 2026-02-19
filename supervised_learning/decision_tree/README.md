&lt;br&gt;
&lt;br&gt;

## Resources
&lt;br&gt;
**Read or watch**:
&lt;br&gt;
&lt;ul&gt;

&lt;li&gt;(1) [Rokach and  Maimon (2002) : Top-down induction of decision trees classifiers : a survey](/rltoken/ZO16F7fr9NvfeOjEnp8B7Q)&lt;/li&gt;
&lt;li&gt;(2) [Ho et al.  (1995) : Random Decision Forests](/rltoken/__GOgr6LnPabR6CZrMzhEA)&lt;/li&gt;
&lt;li&gt;(3) [Fei et al.  (2008) : Isolation forests](/rltoken/HaUBckXPCUf-m3XQJjd2UA)&lt;/li&gt;
&lt;li&gt;(4) [Gini and Entropy clearly explained : Handling Continuous features in Decision Trees](/rltoken/4pdOx48FlOz0Sk2wjbDWCg)&lt;/li&gt;
&lt;li&gt;(5) [Abspoel and al.  (2021) : Secure training of decision trees with continuous attributes](/rltoken/h6OdoOuWktRws91gA4__kA)&lt;/li&gt;
&lt;li&gt;(6) [Threshold Split Selection Algorithm for Continuous Features in Decision Tree](/rltoken/OUrK87LDp8OF77-dnw7dQw)&lt;/li&gt;
&lt;li&gt;(7) [Splitting Continuous Attribute using Gini Index in Decision Tree](/rltoken/6Bwn1F-iSb0Ri2NAZTJqIw)&lt;/li&gt;
&lt;li&gt;(8) [How to handle Continuous Valued Attributes in Decision Tree](/rltoken/n5N8sxhVKYefEyvEA6MwbQ)&lt;/li&gt;
&lt;li&gt;(9) [Decision Tree problem based on the Continuous-valued attribute](/rltoken/OyJCx7dudwrTZ_6JvOMtJA)&lt;/li&gt;
&lt;li&gt;(10) [How to Implement Decision Trees in Python using Scikit-Learn(sklearn)](/rltoken/jn8RW-HGrQKk60_QX_tplg)&lt;/li&gt;
&lt;li&gt;(11)[Matching and Prediction on the Principle of Biological Classification by William A. Belson](/rltoken/s96-WXoMT-dRn1m2_zphSg)&lt;/li&gt;

&lt;/ul&gt;
 &lt;br&gt;
**Notes** 
&lt;br&gt;
&lt;ul&gt;
&lt;li&gt; This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons. First, it gives us confidence in our skills. Second, it helps us when we need to build our own tools to solve unsolved problems.&lt;/li&gt;
&lt;li&gt; The first three references point to historical papers where the concepts were first studied. &lt;/li&gt;
&lt;li&gt; References 4 to 9 can help if you feel you need some more explanation about the way we split nodes.&lt;/li&gt;
&lt;li&gt; William A. Belson is usually credited for the invention of decision trees (read reference 11).&lt;/li&gt;
&lt;li&gt;  Despite our efforts to make it efficient, we cannot compete with Sklearn&#39;s implementations (since they are done in C). In real life, it is thus recommended to use Sklearn&#39;s tools. &lt;/li&gt;
&lt;li&gt; In this regard, it is warmly recommended to watch the video referenced as (10) above. It shows how to use Sklearn&#39;s decision trees and insists on the methodology. &lt;/li&gt;
&lt;/ul&gt;
&lt;br&gt;

## Tasks
&lt;br&gt;
&lt;p&gt;We will progressively add methods in the following 3 classes :&lt;/p&gt;

&lt;pre&gt;&lt;code style=&quot;font-size:10px&quot;&gt;class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature                  = feature
        self.threshold                = threshold
        self.left_child               = left_child
        self.right_child              = right_child
        self.is_leaf                  = False
        self.is_root                  = is_root
        self.sub_population           = None    
        self.depth                    = depth
                
class Leaf(Node):
    def __init__(self, value, depth=None) :
        super().__init__()
        self.value   = value
        self.is_leaf = True
        self.depth   = depth

class Decision_Tree() :
    def __init__(self, max_depth=10, min_pop=1, seed=0,split_criterion=&quot;random&quot;, root=None) :
        self.rng               = np.random.default_rng(seed)
        if root :
            self.root          = root
        else :
            self.root          = Node(is_root=True)
        self.explanatory       = None
        self.target            = None
        self.max_depth         = max_depth
        self.min_pop           = min_pop
        self.split_criterion   = split_criterion
        self.predict           = None
		&lt;/code&gt;&lt;/pre&gt;
&lt;br&gt;		
&lt;ul&gt;
&lt;li&gt; Once built, decision trees are binary trees : a node either is a leaf or has two children. It never happens that a node for which `is_leaf` is `False` has its `left_child` or `right_child` left unspecified.&lt;/li&gt;
&lt;li&gt; The first three tasks are a warm-up designed to review the basics of class inheritance and recursion (nevertheless, the functions coded in these tasks will be reused in the rest of the project).&lt;/li&gt;
&lt;li&gt; Our first objective will be to write a `Decision_Tree.predict` method that takes the explanatory features of a set of individuals and returns the predicted target value for these individuals.&lt;/li&gt;
&lt;li&gt; Then we will write a method `Decision_Tree.fit` that takes the explanatory features and the targets of a set of individuals, and grows the tree from the root to the leaves to make it in an efficient prediction tool.&lt;/li&gt;
&lt;li&gt; Once these tasks will be accomplished, we will introduce a new class `Random_Forest` that will also be a powerful prediction tool.&lt;/li&gt;
&lt;li&gt; Finally, we will write a variation on `Random_Forest`, called `Isolation_Random_forest`, that will be a tool to detect outliers.&lt;/li&gt;
&lt;/ul&gt;
&lt;br&gt;

## Requirements
&lt;br&gt;
&lt;ul&gt;
&lt;li&gt;&lt;b&gt;You should carefully read all the concept pages attached above.&lt;/b&gt;&lt;/li&gt;
&lt;li&gt;All your files will be interpreted/compiled on Ubuntu 20.04 LTS using &lt;code&gt;python3&lt;/code&gt; (version 3.9)&lt;/li&gt;
&lt;li&gt;Your files will be executed with &lt;code&gt;numpy&lt;/code&gt; (version 1.25.2)
&lt;li&gt;All your files should end with a new line&lt;/li&gt;
&lt;li&gt;Your code should use the `pycodestyle` style (version 2.11.1)&lt;/li&gt;
&lt;li&gt;The first line of all your files should be exactly &lt;code&gt;#!/usr/bin/env python3&lt;/code&gt;&lt;/li&gt;
&lt;li&gt;A &lt;code&gt;README.md&lt;/code&gt; file, at the root of the folder of the project, is mandatory&lt;/li&gt;
&lt;li&gt;All your modules should have documentation &lt;code&gt;(`python3 -c &#39;print(__import__(&quot;my_module&quot;).__doc__)&#39; `)&lt;/code&gt;&lt;/li&gt;
&lt;li&gt;All your classes should have documentation &lt;code&gt;(`python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.__doc__)&#39; `)&lt;/code&gt;&lt;/li&gt;
&lt;li&gt;All your functions (inside and outside a class) should have documentation &lt;code&gt;(`python3 -c &#39;print(__import__(&quot;my_module&quot;).my_function.__doc__)&#39; ` &lt;/code&gt; and &lt;code&gt; `python3 -c &#39;print(__import__(&quot;my_module&quot;).MyClass.my_function.__doc__)&#39; `) &lt;/code&gt;&lt;/li&gt;
&lt;li&gt;All your files must be executable&lt;/li&gt;
&lt;/ul&gt;
