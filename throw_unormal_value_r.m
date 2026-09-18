function [x,n]=throw_unormal_value_r(x,std_threshold) 
 %
 flag=0;n=0; 
 while (flag==0) 
     n=n+1; 
     if n>10 
         break; 
     end 
  a=find(x<(mean(x)-std_threshold*std(x))); 
  for k=1:length(a) 
      if a(k)==1 
          x(a(k))=x(a(k)+1); 
      elseif a(k)==length(x) 
          x(a(k))=x(a(k)-1); 
      else 
          x(a(k))=(x(a(k)-1)+x(a(k)+1))/2; 
      end 
  end 
   
  b=find(x>(mean(x)+std_threshold*std(x))); 
  for k=1:length(b) 
      if b(k)==1 
          x(b(k))=x(b(k)+1); 
      elseif b(k)==length(x) 
          x(b(k))=x(b(k)-1); 
      else 
          x(b(k))=(x(b(k)-1)+x(b(k)+1))/2; 
      end 
  end 
   
  if ((length(a)==0)&(length(b)==0)) 
      flag=1; 
  end 
   
end 

