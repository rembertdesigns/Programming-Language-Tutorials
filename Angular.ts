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


// MODULES

import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AppComponent } from './app.component';

@NgModule({
  declarations: [
    AppComponent,
  ],
  imports: [
    CommonModule,
    BrowserModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})

export class AppModule {}


// SERVICES 

  // in a model ts file
  export class ModelName {
    id: number;
    date: Date;
    weight: number;
  }

  // in a .service.ts file
  import { Injectable } from "@angular/core";
  import { ModelName } from "./modelName";

  @Injectable({
    providedIn: 'root???'
  })
  export class NameService {
    public arr: ModelName[] = [
      {id:1, date:new Date(), weight:75},
      {id:2, date:new Date(), weight:80},
      {id:3, date:new Date(), weight:85},
    ]

    constructor(
      public namesSvc: NameService
    ) {}
  }


// INPUT (parent to child)

  // .html (parent)
  @component({ template: `
    <child-component [show]="show"></child-component>
  `})

  // .component.ts (child)
  import { Component, OnInit, Input } from '@angular/core';
  @Component({})
  export class ChildrenComponent implements OnInit {
    @Input() show: boolean;
  }


// OUTPUT (child to parent)

  // .component.ts (child)
  import { Component, OnInit, Output, EventEmitter } from '@angular/core';
  @Component({})
  export class ChildrenComponent implements OnInit {
    @Output() create = new EventEmitter();
    constructor() {}
    public createSmth(): void {
      this.create.emit("Any data you want in your parent");
    }
  }

  // .html (parent)
  `<child-component (create)="createSmthElse($event)"></child-component>`

  // .component.ts (parent)
  public createSmthElse(line: string): void {
    console.log(line);
  }


// VIEWCHILD (bind HTML elements with Angular elements)

  // .component.ts
  import { AfterViewInit, Component, ViewChild } from '@angular/core';
  @Component({})
  export class ChildrenComponent implements AfterViewInit {
    @ViewChild('htmlElement') htmlElement: any;
    ngAfterViewInit() {
      console.log(this.htmlElement);
    }
  }

  // .html
  `<any-element #htmlElement></any-element>`


  // HTML

  // inline js 
  `<p>{{ namesSvc.arr[0].id }}</p>`
  `<p>{{ namesSvc.arr[0].date | date:'dd/MM/yyyy' }}</p>`
  `<p>{{ "lowercase" | uppercase }}</p>`

  // events
  `<div (click)="toggleShow()"></div>`
  `<p [hidden]="!show"></p>`
  `<p *ngIf="show"></p>`

  // loops
  `<p *ngFor="let name of namesSvc.arr; index as i">{{ name.weight }}</p>`

  // child component
  `<div>
    <child-component></child-component>
  </div>`

  // attributes
  [ngClass]="{'my-class': true}"
  [ngClass]="{'class-1': true, 'class-2': true}"
  [ngClass]="true ? 'class-1' : 'class-2'"
  [ngClass]="[true ? 'class-1' : 'class-2', true ? 'class-3 : class-4']"
  [src]=""
  [alt]=""
  [anything]=""