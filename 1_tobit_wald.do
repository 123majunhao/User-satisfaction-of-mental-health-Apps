import delimited D:\data1.csv, varnames(1) clear   //录入数据
tobit pd x1-x13,ll(0) ul(4)   //tobit回归,ll(#)下边界，ul(#)上边界 x1-x13是自变量 变量区分大小写
est store m1   //储存回归结果
tobit nd x1-x13,ll(0) ul(4)
est store m2 
suest m1 m2   //显示两个回归结果
testnl abs([m1_pd]x1)=abs([m2_nd]x1)   //Wald检验
testnl abs([m1_pd]x2)=abs([m2_nd]x2)
testnl abs([m1_pd]x3)=abs([m2_nd]x3)
testnl abs([m1_pd]x4)=abs([m2_nd]x4)
testnl abs([m1_pd]x5)=abs([m2_nd]x5)
testnl abs([m1_pd]x6)=abs([m2_nd]x6)
testnl abs([m1_pd]x7)=abs([m2_nd]x7)
testnl abs([m1_pd]x8)=abs([m2_nd]x8)
testnl abs([m1_pd]x9)=abs([m2_nd]x9)
testnl abs([m1_pd]x10)=abs([m2_nd]x10)
testnl abs([m1_pd]x11)=abs([m2_nd]x11)
testnl abs([m1_pd]x12)=abs([m2_nd]x12)
testnl abs([m1_pd]x13)=abs([m2_nd]x13)
/* 参考：https://blog.csdn.net/weixin_39825322/article/details/113034025 */
