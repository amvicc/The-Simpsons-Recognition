(this.webpackJsonpfrontend=this.webpackJsonpfrontend||[]).push([[0],{58:function(e,t,a){e.exports=a(86)},63:function(e,t,a){},64:function(e,t,a){},86:function(e,t,a){"use strict";a.r(t);var n=a(0),i=a.n(n),r=a(9),l=a.n(r),c=(a(63),a(64),a(11)),s=a(45),o=a(46),m=a(25),d=a(49),h=a(48),p=a(47),u=a.n(p),g=a(128),b=a(129),f=a(122),j=a(126),v=a(120),E=function(e){Object(d.a)(a,e);var t=Object(h.a)(a);function a(e){var n;return Object(s.a)(this,a),(n=t.call(this,e)).state={imageURL:"",image:null,name:" ",isLoading:!1},n.handleUploadImage=n.handleUploadImage.bind(Object(m.a)(n)),n.handleChange=n.handleChange.bind(Object(m.a)(n)),n}return Object(o.a)(a,[{key:"handleUploadImage",value:function(e){var t=this;e.preventDefault();var a=new FormData;a.append("file",this.state.image),this.setState(Object(c.a)(Object(c.a)(Object(c.a)(Object(c.a)({},this.state.image),this.state.imageURL),this.state.name),{},{isLoading:!0})),u()({method:"POST",url:"/upload",data:a}).then((function(e){var a=e.data;a=a.replace("_"," ");var n="This is "+a.toLowerCase().split(" ").map((function(e){return e.charAt(0).toUpperCase()+e.slice(1)})).join(" ");t.setState(Object(c.a)(Object(c.a)(Object(c.a)({},t.state.image),t.state.imageURL),{},{name:n,isLoading:!1}))})).catch((function(e){t.setState(Object(c.a)(Object(c.a)(Object(c.a)(Object(c.a)({},t.state.image),t.state.imageURL),t.state.name),{},{isLoading:!1})),alert("Server error:"+e)}))}},{key:"handleChange",value:function(e){void 0!==e.target&&this.setState(Object(c.a)({imageURL:URL.createObjectURL(e.target.files[0]),image:e.target.files[0]},this.state.name))}},{key:"render",value:function(){return i.a.createElement(g.a,{width:"100%",style:{margin:75}},this.state.isLoading?i.a.createElement(v.a,null):i.a.createElement(f.a,{container:!0,spacing:3,direction:"row",justify:"center",alignItems:"center"},i.a.createElement(f.a,{item:!0,xs:3},i.a.createElement("input",{onChange:this.handleChange,type:"file",id:"contained-button-file",style:{display:"none"}}),i.a.createElement("label",{htmlFor:"contained-button-file"},i.a.createElement(b.a,{style:{marginTop:"25px",width:"315px"},variant:"contained",color:"primary",component:"span"},"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c \u0444\u043e\u0442\u043e\u0433\u0440\u0430\u0444\u0438\u044e \u0421\u0438\u043c\u043f\u0441\u043e\u043d\u0430")),i.a.createElement(b.a,{style:{marginTop:"25px",width:"315px"},variant:"contained",color:"primary",onClick:this.handleUploadImage},"\u0423\u0437\u043d\u0430\u0442\u044c \u0438\u043c\u044f \u0421\u0438\u043c\u043f\u0441\u043e\u043d\u0430")," "===this.state.name?i.a.createElement("div",null):i.a.createElement(j.a,{id:"standard-read-only-input",value:this.state.name,InputProps:{readOnly:!0},variant:"outlined",style:{marginTop:"25px",width:"315px",textAlign:"center"}})),i.a.createElement(f.a,{item:!0,xs:3},""===this.state.imageURL?i.a.createElement("div",null):i.a.createElement("img",{style:{width:"400px"},src:this.state.imageURL,alt:"img"}))))}}]),a}(i.a.Component),O=(a(85),a(123)),y=a(124),L=a(125);var U=function(){return i.a.createElement("div",{className:"App"},i.a.createElement(O.a,{position:"static"},i.a.createElement(y.a,{style:{alignItems:"center",justifyContent:"center"}},i.a.createElement(L.a,{variant:"h6"},"\u0420\u0430\u0441\u043f\u043e\u0437\u043d\u043e\u0432\u0430\u043d\u0438\u0435 \u0421\u0438\u043c\u043f\u0441\u043e\u043d\u043e\u0432"))),i.a.createElement(E,null))};l.a.render(i.a.createElement(i.a.StrictMode,null,i.a.createElement(U,null)),document.getElementById("root"))}},[[58,1,2]]]);
//# sourceMappingURL=main.89b4aaa3.chunk.js.map