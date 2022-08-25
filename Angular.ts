// ANGULAR - Javascript Framework - by Richard Rembert



/**/
// npm install @angular/cli -g
// npm i @angular/cli -g
// ng new <projectName>
// npm install
// npm install tslint
// npm install --save json-server
// ng generate component <componentName>
// ng g c <componentName>
// ng g c <folder>/<componentName>
// ng g service <serviceName>
// ng serve
// npm start
// npm run <custom-script from package.json>
// ng build
// ng build --prod
// prod files are now in dist/<projectName>
// more cli commands -> https://malcoded.com/posts/angular-fundamentals-cli/


// Angular cheat sheet -> https://angular.io/guide/cheatsheet


// COMPONENTS

import { Component, ViewEncapsulation } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'name-component', // <name-component> in HTML
  encapsulation: ViewEncapsulation.None, // disables encapsulation (shadow dom) -> css is applied everywhere
  templateUrl: './name.component.html',
  styleUrls: ['./name.component.css']
})

export class NameComponent {
  public pretrad: string;
  public show: boolean;

  constructor (
    private router: Router
  ) {
    this.pretrad = 'MODULES.NAME.';
    this.show = true;
  }

  public ngOnInit(): void {}

  public toggleShow(): void {
    this.show = !this.show;
  }
}