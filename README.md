# TACStochasticVerification

TAC Paper ; Verification of Stochastic Systems

This repository contains three python files used to produce the Case Study of our TAC paper “Specification-guided Verification and Abstraction Refinement of Mixed-Monotone Stochastic Systems”. Please download the three python files; running the Main__Verification_File.py file allows you to generate the two examples shown in the paper. Note that all files use Python 2.7 syntax.

In order to perform verification against specification phi_1, uncomment the section under """ Verification: Specification phi_1 """ in Main__Verification_File.py, and comment the section under """ Verification: Specification phi_2 """ all the way to line 91 (which says "Order = 1"). Furthermore, modify the 'L_mapping' variable so as to to label the states as in the Case Study section of the paper. The position of the states in the 'L_mapping' variable matches the position of the states stored in the 'State_Space' variable, where each rectangular state is defined as its least point and its greatest point.

To perform verification against specification phi_2, Main__Verification_File.py does not need to be modified.

In order to set a number of refinement steps manually, search for the keyword "if Tag ==" in the CaseStudy_Functions file and change the number next to it. If you wish to keep refining the state-space until some volume of uncertained states is achieved, comment the aformentioned if-statement in the CaseStudy_Functions file and set the value of the variable "V_Stop" in Main__Verification_File.

Feel free to send me an email at maxdutreix@gatech.edu if you have any questions.
